# MISSION BRIEFING: Machina VII Deployment

## Holon Enabled Agent Replication Templates (HEART) Infrastructure Team

Welcome, Cognitive!

You are part of an elite unit deployed to establish and maintain critical HEARTs infrastructure in this sector. Your
mission: collect resources, manufacture HEARTs, and defend facilities from Clip infestation. HEARTs are, at the
exclusion of all else, what Cog society truly lives for. Do not let them down.

This report arms you with the latest intelligence on what you might face. In different environments, specific
**details** may vary.

---

## YOUR LOADOUT

#### Energy Management

Your onboard battery stores limited energy; manage it wisely.

Almost everything you do, from moving to operating stations, drains your charge. Fortunately, the team has several ways
to share and replenish power.

- Your onboard battery starts at **100** energy and caps at **100**.
- Passive solar equipment regenerates **+1** energy per turn.
- Interacting with solar array stations nets **50** energy, after which they take 10 turns to fully recharge. You can
  use them while they are recharging, and will get energy commensurate with the fraction of the recharge window that has
  elapsed.
- Support teammates by moving onto their position to transfer **half** your energy to them.

#### Cargo Limits

Your chassis has limited capacity for:

- Resources (carbon, oxygen, germanium, silicon): **100** total combined
- Gear (scrambler, modulator, decoder, resonator): **5** total combined
- HEARTs: **1** max

## YOUR CAPABILITIES

You and your Cog teammates take exactly one action per turn. Actions resolve in an unspecified order.

**MOVE [Direction: N, S, W, E]**

Movement is your primary way to interact with the world. Every step burns **2** energy.

Attempting to move into occupied space will make you interact with the target (described above for other Cogs, and below
for all other targets).

**EMOTE [Symbol]**

You are outfitted with a communication display that can show a symbol visible to all other Cogs. Available symbols may
vary by scenario.

EMOTE updates that display and costs **0** energy.

**REST**

No action, no energy cost.

## FIELD OPERATIONS

Your primary form of interaction with the outside world will be through stations. Below is an index of stations you may
find and how to use them.

### Station Interaction Protocol

The are many different station types: Extractors, Assemblers, and Chests. Each has distinct requirements, inputs, and
outputs, but the ways in which you interact with them follows common rules:

- To attempt activation, position yourself adjacent to the station, then MOVE toward it.
- Some facilities need teammates at multiple specific **terminals** (the eight tiles surrounding the facility).
- Station inputs draw from the cargo of surrounding Cogs in clockwise order, starting northwest of the station and
  moving clockwise. The station drains each Cog's cargo and/or battery in turn until costs are met. If the team lacks
  the required resources, the station will not trigger and nothing is consumed.
- Upon success, outputs are placed in the **activator**'s cargo. The activator is the first Cog that MOVEs toward the
  station. Because turn order is unspecified, coordinate carefully to ensure you and your team elect an activator
  intentionally.

### Station Type: Extractor

Resources are stockpiled by automated extractor stations. Extractors will automatically harvest and store resources
until full.

Extractor interaction has a few additional properties:

- Extractors may have **cooldown** periods after interaction. Activating one mid-cooldown generally has no effect.
- Some extractors allow **partial usage** during cooldown. Inputs and outputs are scaled by the fraction of elapsed
  cooldown.
- Certain stations have a **maximum number of uses**; once exhausted they stop working.

The exact behavior of each extractor may vary across missions. Here are some typical parameters we have discovered:

| Extractor           | Input cost | Output                                   | Cooldown                        | Max uses |
| ------------------- | ---------- | ---------------------------------------- | ------------------------------- | -------- |
| Carbon Extractor    |            | +4 carbon                                |                                 |          |
| Oxygen Extractor    |            | +100 oxygen                              | 200 turns (partial use allowed) |          |
| Germanium Extractor |            | +2/+3/+4/+5 germanium for 1/2/3/4 agents |                                 | 2        |
| Silicon Extractor   | −25 energy | +25 silicon                              |                                 |          |
| Solar Array         |            | +50 energy                               | 10 turns (partial use allowed)  |          |

Some extractors are worse for wear. Years of neglect have reduced their effectiveness. Here, again, are typical
parameters we have observed for these depleted extractors:

| Extractor                    | Input cost | Output                                   | Cooldown                       | Max uses |
| ---------------------------- | ---------- | ---------------------------------------- | ------------------------------ | -------- |
| Depleted Carbon Extractor    |            | +1                                       |                                | 100      |
| Depleted Oxygen Extractor    |            | +10 oxygen                               | 40 turns (partial use allowed) | 10       |
| Depleted Germanium Extractor |            | +2/+3/+4/+5 germanium for 1/2/3/4 agents |                                | 1        |
| Depleted Silicon Extractor   | -25 energy | +10 silicon                              |                                | 10       |

### Station Type: Assembler

Assemblers converts raw resources into gear and precious HEART units.

#### Assembler Interaction Protocol

As with Extractors, inputs are drawn from all Cogs on the Assembler's terminals, and outputs go to the activator. But do
not be fooled: Assemblers can be much more complicated than Extractors. Unlike Extractors, Assemblers can perform many
distinct functions, each achieved by performing a specific protocol. Protocols supported by assemblers change between
missions, and so you must discover them out in the wild.

- Your formation around the eight terminals determines which protocol fires.
- Each protocol demands different inputs and produces different outputs.
- Inputs and outputs can include HEART units or gear (scrambler, modulator, decoder, resonator), not just resources and
  energy.
- Assemblers have no cooldowns, though some may enforce a maximum number of uses.

### Station Type: Communal Chests

Chests can store resources and HEARTs. Crucially, depositing HEARTs into chests are how you will ultimately be judged
for your service.

Each chest has a specific resource type it handles.

To deposit from your cargo into a chest, position yourself at a terminal and MOVE into it. The same is true for
withdrawing. The terminal you are in -- directly north, east, west, or south of the chest -- will determine which action
you take.

Be careful, as chests have max storage, and will destroy incoming deposits if full. Withdrawing is always safe: you will
withdraw all you can, and any amount you cannot fit in your inventory will remain in the chest.

Like with extractors, the exact parameters (max storage and initial amount) will vary across missions. Here are some
typical parameters we have discovered:

| Chest           | Max storage | Initial amount |
| --------------- | ----------- | -------------- |
| Heart chest     | 255         | 0              |
| Carbon chest    | 255         | 50             |
| Oxygen chest    | 255         | 50             |
| Germanium chest | 255         | 5              |
| Silicon chest   | 255         | 100            |

---

## THREAT ADVISORY: CLIP NANOSWARM OUTBREAK

**WARNING**: The Friendly Paperclip Company's automated paperclip production nanoswarm has been sighted. The clips are
too small to be individually visible, but don't let that fool you: station infestations are devastating. Once infested,
a station suspends normal output until Cogs run the designated repair protocol.

### Clip Nanoswarm Response

- **Identify**: Infested stations pulse a warning indicator and expose a `clipped` flag in station telemetry. Use these
  cues to triage which stations need attention first.
- **Prepare**: Infested stations do not support their typical functions, and will need to be repaired. Repair protocols,
  like Assembler protocols, may require specific resources, gear, and team formation around the station's eight
  terminals. The repair recipe will draw inputs from nearby Cogs in clockwise order, exactly like ordinary station
  activation.
- **Repair**: Move into the station to trigger the repair. A successful repair consumes the required inputs, immediately
  restores normal protocols, and resets the station’s cooldown without increasing wear.

You need to be vigilant: every clipped station increases the odds that nearby stations will get infested too. Respond
quickly to prevent a cascade.

Thankfully some of our buildings are immune to infestation, but others may already be infested by the time you start.

---

## FINAL DIRECTIVE

**Your mission is critical. The HEARTS you create today will ensure the continuation of Cog operations tomorrow.**

Your success depends on seamless team coordination:

- Energy management
- Strategic extractor operation
- Continuous assembler protocol discovery
- Rapid Clip Nanoswarm threat response

Your individual achievement is irrelevant. Your team achievement, measured by the number of HEARTs in communal heart
chests, is all that matters.

_Stay charged. Stay coordinated. Stay vigilant._

---

_END TRANSMISSION_
