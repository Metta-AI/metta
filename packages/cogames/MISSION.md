# MISSION BRIEFING: Machina VII Deployment

## Holon Enabled Agent Replication Templates (HEART) Infrastructure Team

### Welcome, Cognitive!

As you know, every Cog wants nothing more than to enjoy in the collective HEARTs collected by the colony. You are part
of an elite unit of Cogs deployed to establish and maintain critical HEARTs infrastructure in this sector. Your mission:
collect resources, manufacture HEARTs, and defend facilities from Clip infestation.

This report arms you with the latest information on what you might face. In different environments, you may find that
some of these **details** vary.

---

## YOUR LOADOUT

#### Energy Management

Your onboard battery stores limited energy -- manage it wisely!

Almost everything you do, from moving to interacting with stations, will expend your energy, but you and the team have
various tools at your disposal to share and replenish your stores.

- Your onboard battery starts with **100** energy and stores a maximum of **100**.
- You are equipped with passive solar equipment. It regenerates **+1** energy/turn
- You can interact with solar array stations, collecting **50** energy with **10**-turn cooldown
- Support your teammates: move into an ally's position to transfer **half** your onboard battery's energy to theirs

#### Cargo Limits

Your chassis has limited capacity for:

- Resources (carbon, oxygen, germanium, silicon): **100** total combined
- Gear (scrambler, modulator, decoder, resonator): **5** total combined
- HEARTs: **1** max

## YOUR CAPABILITIES

You and your Cog teammates can take one of these actions per turn. Your actions will execute in an unspecified order.

**MOVE [Direction: N, S, W, E]**

Movement is the main way you can interact with the world. Every step takes one full turn and exhausts **2** energy.

You can move into:

- Empty space: move to position
- Occupied space: interact with the target:
  - Another Cog → Transfer **half** of your onboard battery's energy to theirs
  - Station → Activate or operate the facility (see Interaction Protocols below)

**EMOTE [Symbol]**

You are outfitted with a communication display that can show a symbol, visible to all other Cogs. The available symbols
may vary by scenario.

You can opt to EMOTE and update the symbol in your display. It requires **0** energy cost.

**REST**

No action, no energy cost

## STATIONS

You will need to operate stations of many kinds. Below is an index of what stations are available to you.

### Station Interaction Protocol

- Position yourself adjacent to it, then MOVE toward it
- Multi-Cog operations: Some facilities need teammates at specific **terminals** (the 8 positions surrounding the
  facility)
  - Station inputs are drawn from the cargos of the surrounding cogs in clockwise order, starting at the position
    northwest of the station. The station will draw on each cog's cargo and/or battery in turn, exhausting the full
    amount available, until its input costs are met. If there are insufficient resources among the surrounding cogs, the
    station won't activate and won't consume inputs.
  - Upon successful activation, station outputs are placed in the **activator**'s cargo. The activator is the first Cog
    that MOVEs towards the station. Because you and your teammates' actions will execute in an unspecified order, you
    may need to coordinate.
- Stations may have **cooldown** periods after interaction. During a station's cooldown period, activation will by
  default have no effect
- Stations configured with **exhaustion** grow slower over time: after each successful use their cooldown is multiplied
  by `(1 + exhaustion)`

### Station Type: Extractor

Resources are stockpiled by automated extractor stations. Extractors have finite storage capacity, and will
automatically produce and store resources until they are full.

Stations largely follow the rules of the Station Interaction Protocol. There are a few additional things to know:

- When an extractor is on cooldown, **partial usage** may be supported. In such cases, the station's inputs and outputs
  will be scaled by the fraction of its cooldown period that has elapsed, and its its original cooldown period remains
  in effect; it does not get refreshed.
- Stations may become **depleted**, reducing their efficiency
- Some stations support **limited uses**, after which interaction will have no effect.

Below are default settings for the known extractors, but remember that they can vary by scenario.

| Station                            | Input cost | Output                                   | Cooldown                        | Max uses |
| ---------------------------------- | ---------- | ---------------------------------------- | ------------------------------- | -------- |
| Carbon Extractor (`C`)             |            | +4 carbon                                |                                 |          |
| Oxygen Extractor (`O`)             |            | +100 oxygen                              | 200 turns (partial use allowed) |          |
| Germanium Extractor (`G`)          |            | +2/+3/+4/+5 germanium for 1/2/3/4 agents |                                 | 2        |
| Silicon Extractor (`S`)            | −25 energy | +25 silicon                              |                                 |          |
| Solar Array (`+`)                  |            | +50 energy                               | 10 turns (partial use allowed)  |          |
| Depleted Carbon Extractor (`c`)    |            | +1                                       |                                 | 100      |
| Depleted Oxygen (`o`)              |            | +10 oxygen                               | 40 turns (partial use allowed)  | 10       |
| Depleted Germanium Extractor (`g`) |            | +2/+3/+4/+5 germanium for 1/2/3/4 agents |                                 | 1        |
| Depleted Silicon (`s`)             | -25        | +10 silicon                              |                                 | 10       |

### Station Type: Assembler

This is your primary objective facility! The Assembler converts raw resources into gear and precious HEART units.

#### Assembler Interaction Protocol

You must discover what each assembler is capable of. Assembler protocols change between missions!

- Like extractors:
  - The first Cog that MOVEs into the Assembler activates it
  - Resources are consumed from the team (in the same way)
  - Outputs (gear or hearts) go to the activator
- Unlike extractors:
  - The precise formation of you and your teammates around the terminals of the Assembler determines which Protocol will
    be activated.
  - Different protocols require different inputs and create different outputs
  - Inputs and outputs can include HEART units or gear (scrambler, modulator, decoder, resonator), not just resources
    and energy
  - Some assemblers will require gear to activate but will not consume that gear as an input cost
  - Assemblers do not have cooldowns or a maximum number of uses, and do not get depleted
  - Protocol availability may change based on assembler status [[check: how does status change?]]

### Station Type: Chests

Chests can store resources and HEARTs. Crucially, depositing HEARTs into the communal

**To Deposit**: Position yourself at specific terminals and move into the Chest **To Withdraw**: Position yourself at
different terminals and move into the Chest

- Experiment to discover which positions trigger deposits vs withdrawals
- Primary storage for completed HEARTS units

---

## THREAT ADVISORY: FRIENDLY PAPERCLIP COMPANY OUTBREAK

**WARNING**: The Friendly Paperclip Company's automated paperclip production nanoswarm has been sighted infesting our
outlying extractors. The Clips are actively attempting to infest our facilities and subvert them for optimal paperclip
production.

### Clip Response Actions

- **Identify**: Clipped facilities have altered appearance
- **Prepare**: Reclaiming buildings from the Clips requires special tools, which can be produced at the Atomic
  Assembler. Make sure you have what you need. Some clipped facilities may require more than one Cog and more than one
  tool.
- **Approach**: Move adjacent to the infested facility with your fellow Cogs and explore to find the reclamation
  protocol
- **Reclaim**: Move into the facility to attempt a reclamation protocols

- Act quickly: infestation levels may rise over time, requiring more difficult reclamation procedures, and the Clips
  will drain more and more resources from infected facilities over time!

### Infestation Patterns

- Spreads primarily between nearby extractors
- Infestation rate may accelerate over time or come in waves
- Left unchecked, Clips will fully consume all resources from an extractor and its deposit, leaving only a husk

---

## FINAL DIRECTIVE

**Your mission is critical. The HEARTS you create today will ensure the continuation of Cog operations tomorrow.**

Your individual achievement is irrelevant. Your team achievement, measured by the number of HEARTs in the communal heart
chest, is all that matters.

Your success depends on:

- Efficient energy management
- Strategic facility operations
- Rapid Clip threat response
- Continuous protocol discovery
- Seamless team coordination

_Stay charged. Stay coordinated. Stay vigilant._

---

_END TRANSMISSION_
