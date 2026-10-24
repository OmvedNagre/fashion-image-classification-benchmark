# Docs Index

The `docs/` folder is the source of truth for this project. Code follows docs; when they diverge,
the doc is updated in the same commit and the reason is logged in `decisions.md`.

| File | Answers | Update when |
|---|---|---|
| [prd.md](prd.md) | What are we building, for whom, and what's out of scope? | Scope changes (needs approval) |
| [architecture.md](architecture.md) | What are the parts and where does code live? | Components, stack, or folder layout change |
| [system-design.md](system-design.md) | How does it behave at runtime, under load, and when things fail? | Flows, concurrency, caching, or failure handling change |
| [data-formats.md](data-formats.md) | What files does the pipeline save (metrics JSON, summary.csv, caches, splits) and what is inside them? | Before changing any saved file format |
| [module-interfaces.md](module-interfaces.md) | What do the main `src/` functions take and return? | Before changing a shared function signature |
| [decisions.md](decisions.md) | Why is it built this way? | Every non-trivial choice, as it's made |
| [progress.md](progress.md) | Where are we right now? | Every session and phase transition |
| [phases/](phases/) | What exactly gets built in each phase, grouped by release (`v1/`, `v2/`, …)? | Phase start, task completion, phase close |

## Architecture vs system design
- **architecture.md** is the map: components, boundaries, stack, folder structure. Mostly static.
- **system-design.md** is the movie: what happens when a request arrives, two users race for the
  same resource, a service goes down, or traffic spikes.

## Phase files
`phases/vN/phase-NN-<slug>.md`, created from `phases/_phase-template.md`. `vN` is the release
(its major version: phases in `v1/` ship as tag `v1.x.y`); numbers are two digits and restart at 01
in each version. A phase id is `vN/NN` (e.g. `v1/02`): run `/phase-start v1/02`, branch
`phase/v1-02-<slug>`. Each phase leaves the app working and demoable.
