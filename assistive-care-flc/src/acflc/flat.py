"""The assistive-care flat controller: variables, membership functions, rules.

Scenario
--------
One room of a flat occupied by a resident with a physical disability that
limits mobility. The controller holds thermal comfort and lighting, which are
the two environmental services a resident with restricted movement can least
easily correct for themselves: someone who cannot readily cross the room to
reach a thermostat or a light switch, or who cannot quickly add or remove a
layer of clothing, depends on the room getting it right unprompted.

That framing is what motivates the two inputs beyond the obvious sensors.

`activity` matters because thermoregulation is not uniform across residents.
Reduced mobility means reduced metabolic heat production, and several
conditions affecting mobility -- spinal cord injury in particular -- also
impair vasomotor and sweating responses below the level of the lesion. A
resting occupant therefore needs a warmer room than a general-purpose
thermostat would choose, and the shift is large enough to be worth encoding.

`preference` matters because comfort is subjective and, in an assistive
setting, contested: what the building manager considers efficient and what the
resident considers comfortable are not the same quantity. Exposing preference
as a first-class input keeps the resident in the loop rather than optimising
them out of it.

Design decisions recorded here, for the Part 1 justification (7 marks)
----------------------------------------------------------------------
* **Mamdani, not Sugeno.** See `docs/` and the report. Briefly: the rule base
  is elicited from human domain knowledge rather than fitted to logged data,
  and Mamdani's fuzzy consequents keep it legible to the occupational
  therapist or carer who has to sign off on the behaviour. The plant is slow
  -- room air temperature moves over minutes -- so Sugeno's cheaper arithmetic
  buys nothing that matters here.
* **Triangles inside, trapezoids at the edges.** Interior sets need a single
  unambiguous prototype value, which a triangle gives. Edge sets need to
  saturate: 8 degrees and 14 degrees are not meaningfully different to an
  occupant and both should command full heat, which is what a trapezoid with
  a == b expresses and a triangle cannot. Both shapes are piecewise linear and
  so cost far less to evaluate than Gaussians or bell curves, which matters
  once the Part 2 GA is calling the controller millions of times.
* **Overlap at roughly half the base width.** Adjacent sets cross near mu=0.5,
  which is the standard choice: enough overlap that the control surface stays
  continuous and no input value falls through the rule base, little enough
  that a single reading rarely activates more than two sets per variable.
* **Centroid defuzzification.** Discussed in `run_part1.py`, where it is
  compared numerically against bisector and mean-of-maximum.
"""

from __future__ import annotations

from .controller import FuzzyController, Rule
from .membership import MF, Variable

# ---------------------------------------------------------------- input sets

#: Room air temperature. The universe runs well past the comfort band in both
#: directions so that a fault condition (heating failure in winter, solar gain
#: in summer) still lands inside the modelled range rather than being clipped.
ROOM_TEMP = Variable(
    name="room_temp", unit="degC", lo=14.0, hi=34.0,
    mfs=(
        MF("Cold",        "trapmf", (14.0, 14.0, 16.0, 19.0)),
        MF("Cool",        "trimf",  (17.0, 19.5, 22.0)),
        MF("Comfortable", "trimf",  (20.5, 22.5, 24.5)),
        MF("Warm",        "trimf",  (23.0, 25.5, 28.0)),
        MF("Hot",         "trapmf", (26.0, 29.0, 34.0, 34.0)),
    ),
)

#: Occupant activity index, 0-10, fused from a passive infrared motion sensor
#: and a wrist-worn accelerometer. 0 is asleep or fully still; 10 is sustained
#: effortful movement such as a transfer or self-propelling a manual chair.
ACTIVITY = Variable(
    name="activity", unit="index", lo=0.0, hi=10.0,
    mfs=(
        MF("Resting", "trapmf", (0.0, 0.0, 1.5, 3.5)),
        MF("Light",   "trimf",  (2.5, 5.0, 7.5)),
        MF("Active",  "trapmf", (6.5, 8.5, 10.0, 10.0)),
    ),
)

#: Daylight reaching the room, as a percentage of the design task illuminance.
#: Expressed as a percentage rather than raw lux so that the same controller
#: transfers between rooms with different glazing without retuning.
DAYLIGHT = Variable(
    name="daylight", unit="% of design lux", lo=0.0, hi=100.0,
    mfs=(
        MF("Dark",   "trapmf", (0.0, 0.0, 10.0, 30.0)),
        MF("Dim",    "trimf",  (20.0, 45.0, 70.0)),
        MF("Bright", "trapmf", (60.0, 80.0, 100.0, 100.0)),
    ),
)

#: Standing comfort preference on a -5 (prefers cooler) to +5 (prefers warmer)
#: dial, set by the resident and persisted. This is the resident's authority
#: over the controller, not a transient request.
PREFERENCE = Variable(
    name="preference", unit="scale", lo=-5.0, hi=5.0,
    mfs=(
        MF("Cooler",  "trapmf", (-5.0, -5.0, -3.0, -1.0)),
        MF("Neutral", "trimf",  (-2.0, 0.0, 2.0)),
        MF("Warmer",  "trapmf", (1.0, 3.0, 5.0, 5.0)),
    ),
)

# --------------------------------------------------------------- output sets

#: Signed HVAC command: negative is cooling power, positive is heating power,
#: both as a percentage of plant capacity. A single signed actuator is used
#: rather than separate heat and cool outputs because the two are physically
#: exclusive -- a controller that can command both at once can fight itself,
#: and a signed axis makes that state unrepresentable.
HVAC = Variable(
    name="hvac", unit="% capacity (-cool/+heat)", lo=-100.0, hi=100.0,
    mfs=(
        MF("CoolHigh", "trapmf", (-100.0, -100.0, -80.0, -45.0)),
        MF("CoolLow",  "trimf",  (-70.0, -35.0, 0.0)),
        MF("Off",      "trimf",  (-15.0, 0.0, 15.0)),
        MF("HeatLow",  "trimf",  (0.0, 35.0, 70.0)),
        MF("HeatHigh", "trapmf", (45.0, 80.0, 100.0, 100.0)),
    ),
)

#: Lamp dimmer level as a percentage of full output.
DIMMER = Variable(
    name="dimmer", unit="%", lo=0.0, hi=100.0,
    mfs=(
        MF("Off",    "trapmf", (0.0, 0.0, 5.0, 20.0)),
        MF("Low",    "trimf",  (10.0, 30.0, 50.0)),
        MF("Medium", "trimf",  (40.0, 60.0, 80.0)),
        MF("High",   "trapmf", (70.0, 88.0, 100.0, 100.0)),
    ),
)

# --------------------------------------------------------------- the rulebase

# Thermal rules are generated from an additive fuzzy associative memory (FAM)
# rather than written out one at a time. Each of the three antecedent
# variables contributes an integer shift on a five-point demand scale, the
# shifts are summed and the result is saturated back into range:
#
#     modifier = clip( shift(activity) + shift(preference), -1, +1 )
#     demand   = clip( base(temp) + modifier,               -2, +2 )
#
# The inner clip on the modifier is load-bearing and was added after the
# first draft of this rule base was audited. Without it, activity and
# preference each contribute a full step and can therefore *overrule* the
# temperature reading rather than adjust it: a Comfortable room with a resting
# occupant who prefers warmth summed to +2 and commanded HeatHigh, which would
# drive a 22.5 degree room towards 30. Clamping the combined modifier to a
# single step guarantees that the measured temperature always sets the demand
# to within one step, so the controller can never heat a Hot room or cool a
# Cold one. That is a safety property of the flat, and it is now true by
# construction rather than by hopeful inspection.
#
# The point of the additive structure is that it is *auditable*. A carer can be
# shown three short tables instead of a 45-row list, and can check the claim
# "resting shifts demand one step towards heating" directly, on its own, rather
# than having to trust 45 independently asserted rules to be mutually
# consistent. It also guarantees monotonicity -- the controller can never
# respond to a colder room by heating less -- which is a property worth having
# by construction rather than by inspection.
#
# Saturation is deliberate and physically right: once demand is at full heat
# there is nowhere further to go, so Cold + Resting + Warmer and
# Cold + Resting + Neutral both command HeatHigh.

_TEMP_BASE = {"Cold": +2, "Cool": +1, "Comfortable": 0, "Warm": -1, "Hot": -2}
_ACTIVITY_SHIFT = {"Resting": +1, "Light": 0, "Active": -1}
_PREFERENCE_SHIFT = {"Cooler": -1, "Neutral": 0, "Warmer": +1}
_HVAC_BY_DEMAND = {-2: "CoolHigh", -1: "CoolLow", 0: "Off", +1: "HeatLow", +2: "HeatHigh"}

# Lighting rules follow the same idea on a four-point scale. The occupant's
# activity sets the illuminance the task needs; available daylight is
# subtracted from it, because the lamp only has to make up the shortfall.
#
# One entry is overridden from what the arithmetic gives, and the override is
# the more interesting design decision of the two:
#
#     daylight Dim + activity Resting  ->  Low, not Off
#
# The arithmetic says a resting occupant in partial daylight needs no lamp at
# all. Falls are the dominant injury risk in this population and the dominant
# reason an assistive flat exists, so the controller keeps a floor of
# orientation lighting whenever daylight is not Bright. It is a deliberate
# choice to trade a little energy for a safety margin, and it is recorded here
# rather than buried so that it can be argued with.

_ACTIVITY_NEED = {"Resting": 1, "Light": 2, "Active": 3}
_DAYLIGHT_OFFSET = {"Dark": 0, "Dim": 1, "Bright": 2}
_DIMMER_BY_LEVEL = {0: "Off", 1: "Low", 2: "Medium", 3: "High"}
_LIGHTING_OVERRIDES = {("Dim", "Resting"): "Low"}


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def thermal_rules():
    """The 45 temperature x activity x preference rules."""
    rules = []
    for t in ROOM_TEMP.mf_names:
        for a in ACTIVITY.mf_names:
            for p in PREFERENCE.mf_names:
                modifier = _clip(_ACTIVITY_SHIFT[a] + _PREFERENCE_SHIFT[p], -1, 1)
                demand = _clip(_TEMP_BASE[t] + modifier, -2, 2)
                rules.append(
                    Rule(
                        antecedents={"room_temp": t, "activity": a, "preference": p},
                        consequent=("hvac", _HVAC_BY_DEMAND[demand]),
                        note=f"demand={demand:+d}",
                    )
                )
    return rules


def lighting_rules():
    """The 9 daylight x activity rules, including the safety-floor override."""
    rules = []
    for d in DAYLIGHT.mf_names:
        for a in ACTIVITY.mf_names:
            level = _clip(_ACTIVITY_NEED[a] - _DAYLIGHT_OFFSET[d], 0, 3)
            mf = _DIMMER_BY_LEVEL[level]
            note = f"level={level}"
            if (d, a) in _LIGHTING_OVERRIDES:
                mf = _LIGHTING_OVERRIDES[(d, a)]
                note = f"level={level} overridden to {mf}: fall-risk floor"
            rules.append(
                Rule(
                    antecedents={"daylight": d, "activity": a},
                    consequent=("dimmer", mf),
                    note=note,
                )
            )
    return rules


def build_controller() -> FuzzyController:
    """The Part 1 controller, with hand-designed membership functions."""
    return FuzzyController(
        inputs=[ROOM_TEMP, ACTIVITY, DAYLIGHT, PREFERENCE],
        outputs=[HVAC, DIMMER],
        rules=thermal_rules() + lighting_rules(),
        name="AssistiveCareRoomFLC",
    )


def fam_tables():
    """The rule base as FAM matrices, for tabulation in the report.

    Returns a dict of {title: (row labels, column labels, cell text)}.
    """
    tables = {}
    for p in PREFERENCE.mf_names:
        cells = [
            [
                _HVAC_BY_DEMAND[
                    _clip(_TEMP_BASE[t]
                          + _clip(_ACTIVITY_SHIFT[a] + _PREFERENCE_SHIFT[p], -1, 1), -2, 2)
                ]
                for a in ACTIVITY.mf_names
            ]
            for t in ROOM_TEMP.mf_names
        ]
        tables[f"HVAC | preference = {p}"] = (
            list(ROOM_TEMP.mf_names), list(ACTIVITY.mf_names), cells
        )

    cells = []
    for d in DAYLIGHT.mf_names:
        row = []
        for a in ACTIVITY.mf_names:
            level = _clip(_ACTIVITY_NEED[a] - _DAYLIGHT_OFFSET[d], 0, 3)
            row.append(_LIGHTING_OVERRIDES.get((d, a), _DIMMER_BY_LEVEL[level]))
        cells.append(row)
    tables["Dimmer"] = (list(DAYLIGHT.mf_names), list(ACTIVITY.mf_names), cells)
    return tables
