# Web lab: demonstration and placement preparation

This guide covers the version 3 additions. Read it alongside the original placement PDF, which describes the original PPO and reward-shaping project. All results here are from a simulator, not a physical rocket.

## Start and demonstrate

1. Double-click `Open-Web-Lab.cmd`. Keep its console window open while using the app. The launcher chooses an available local port and opens your browser. Close the console to stop the server.
2. Start on the introduction: explain the eight observations, four discrete actions, feedback through rewards, and the deliberately flawed hovering policy.
3. Enter the lab. Overview compares saved policies. Filter by weather and inspect the number of flights, landing rate, original score, and simulated duration. Export the filtered records as CSV.
4. Open Flight explorer. Search for `hover`, `failed`, a weather condition, or a seed. Play or scrub the reconstructed trajectory and inspect speed, tilt, and engine actions. The drawing is schematic, not a new video rendered by Box2D.
5. Open Failure predictor. Explain why a classifier is a separate model from the PPO controller. Inspect validation selection, held-out results, and the confusion matrix before making claims.
6. Ask the lab: “Why did the hover policy fail?” Then ask “Which pilot has the best landing rate in gusty weather?” Expand the evidence. The first is retrieval plus local generation; the second is a deterministic calculation over saved flights.
7. Open Broader training. Compare the source and fine-tuned policy under identical evaluation seed numbers. Additional training is optional; a completed comparison is included.

The older desktop control panel and launchers still work. Setup on another Windows computer: install Python (tested with 3.12), run `setup.ps1`, then open the web launcher. Internet is needed for dependency installation. Browser assets, simulations, analytics, and prediction run locally. Generated assistant answers additionally require Ollama with `qwen3:1.7b`; install that model using `ollama pull qwen3:1.7b` if desired. When it is unavailable, source retrieval still works and the UI explicitly labels excerpts instead of generated answers. No paid API key is required. This app serves only loopback; it is not configured for public multi-user hosting or GitHub Pages.

## How the pieces connect

`Gymnasium → PPO action → physics step → FlightMetrics → telemetry SQLite → dashboard / outcome classifier`

`Question → exact numerical query OR TF-IDF retrieval → source excerpts → local Ollama generation → answer with source IDs`

`Saved fixed PPO → randomized-weather fine-tuning → separate checkpoint → paired weather evaluation → comparison`

The frontend is HTML, CSS and browser JavaScript. FastAPI serves the page, assets and JSON endpoints. Uvicorn runs the HTTP server. SQLite is embedded storage, so there is no database service to configure. NumPy provides feature calculations. Scikit-learn supplies preprocessing, classifiers, metrics and text retrieval. Joblib stores the fitted classifier. PyPDF reads the placement PDF for retrieval. Gymnasium/Box2D supplies the environment; Stable-Baselines3/PyTorch supplies PPO training and inference. Matplotlib remains responsible for the original experiment reports. Python subprocesses run long jobs without blocking page requests. One job at a time is allowed; its log and stop button are visible.

## Telemetry: what is actually stored

The included library contains 720 flights: four original policies × three weather conditions × 60 episode seeds, 600000–600059. All use deterministic policy actions and the original environment reward, with reward shaping disabled during evaluation. Calm, breezy and gusty use the same settings as the original weather tests. Each policy currently represents one original training seed.

`artifacts/lab/flights.sqlite` contains one row per flight. Summary columns: a flight ID, policy label, weather, episode seed, SHA-256 checkpoint identity, UTC collection time, outcome, original return, number of steps, fuel proxy and hover fraction. A compressed JSON blob stores the full sequence: step, x, y, horizontal speed, vertical speed, angle, angular speed, left contact, right contact, action and original step reward. Step zero is the reset observation; its action is -1 and reward is zero. Later rows record the observation after their action. Observations use Gymnasium scaling; they are not all SI units. One step represents 1/50 simulated second; wall-clock playback and training speed are separate.

The flight ID hashes checkpoint identity, scenario, episode seed and schema version. Recollection skips existing identical flights. A changed checkpoint produces new IDs. “Collect more flights” evaluates available final checkpoints on the configured default seed set; it can add a newly trained policy, but does not duplicate existing flights. For fresh seed numbers, use the telemetry command with `--seed` and `--episodes`. New collected data does not silently retrain the predictor: use its explicit training button.

Fuel is an engine-use proxy, not kilograms. `failed` includes crash/out-of-bounds termination. `timeout` means reaching the step cap. A high score is not equivalent to a successful landing. Filtering outcomes conditions the statistics: selecting only landed flights necessarily displays 100% landing rate. Compare policies with the same weather, seed set, and outcome filter. The 95% Wilson interval estimates uncertainty in a binomial landing proportion, not variation across training seeds. Shared seed numbers help comparisons but weather changes can alter random draws and trajectories.

## Failure predictor: supervised learning without looking into the future

The target is one of `failed`, `landed`, `timeout`. The forecast is made at step 150, or three seconds. Flights ending by that point are excluded: their outcome is already known. There are 714 eligible flights and six excluded early endings in the included dataset.

For x, y, vx, vy, angle and angular speed, compute last value, mean, standard deviation, minimum and maximum over observations 0–150. Add main-engine and side-engine fractions from actions 1–150. This produces 32 features. The input deliberately excludes final return, final length, weather label, policy name, final outcome and observations after step 150. The final outcome is used only as the supervised target.

Split by episode seed group, keeping every policy/weather version of the same seed in one partition. GroupShuffleSplit first holds out 20% of groups for testing, then 25% of remaining groups for validation (approximately 60/20/20 overall). Fit a standard-scaled logistic regression and a random forest on training data only. Choose the model with the highest validation macro F1. Do not refit or select using the test set. The included selected model is a random forest; all split seed lists and per-flight assignments are saved.

Held-out results: accuracy 77.78%, macro F1 0.668, landing Brier score 0.162 on 144 flights. The confusion matrix has actual rows and predicted columns in the order failed/landed/timeout:

| Actual | Predicted failed | Predicted landed | Predicted timeout |
|---|---:|---:|---:|
| Failed | 5 | 18 | 0 |
| Landed | 4 | 73 | 5 |
| Timeout | 0 | 5 | 34 |

This exposes a significant limitation: failure recall is only 5/23 = 21.7%. Accuracy alone conceals it. The predictor is an educational diagnostic, not a reliable crash alarm. Macro F1 weighs the three classes equally; Brier score measures squared probability error for landing, with lower being better. The reliability table compares predicted probability and observed landing frequency. Probabilities are not guaranteed to be calibrated. No threshold was tuned on the test data. Replay labels each flight as train, validation, test or new; an individual training-flight prediction is not independent evidence of performance.

Generalization is to unseen seed groups for the known policies, not unseen algorithms or real rockets. Adding policies, changing feature definitions or repeatedly consulting the test set requires a new evaluation protocol. A policy-held-out split and additional training seeds are useful future research. Model file hashing detects accidental inconsistency with the report; it does not authenticate untrusted files. Only load the repository's locally produced Joblib artifacts.

## RAG: retrieval augmented generation

The assistant reads project documentation, PDF pages, Python source, and original quality-gate facts. It splits text into overlapping 210-word windows with a 150-word stride. Each chunk retains its file/page and source ID. TF-IDF with word unigrams and bigrams builds a sparse index. The question is transformed using the same vectorizer; cosine-like normalized dot products rank chunks. Up to four chunks with similarity above 0.055 become context.

This is lexical retrieval, not neural embeddings or a vector database. That is a valid, small local RAG baseline, but synonyms and complex questions may retrieve weak evidence. Ollama serves the local Qwen3 1.7B model. The prompt asks it to use supplied evidence, admit insufficient support, and cite source IDs. The app shows retrieved excerpts and file/page identifiers so users can inspect the grounding. Invalid source IDs are removed, but supported citations do not prove every generated sentence is correct. Generation failures fall back to labeled excerpts. No evidence produces an explicit “could not find supporting project evidence” answer.

Recognized numerical questions such as landing rate, average score, best pilot and number of flights use deterministic queries and arithmetic over the telemetry library. They are labeled “computed data,” not passed to the language model to invent numbers. “Best” defaults to observed landing rate unless average score is requested; it is not a universal ranking. The assistant does not control the rocket, run arbitrary commands, or modify training from its text response. It indexes local material in memory on first use; restart the server after editing indexed documentation.

## Broader RL training and honest comparison

The experiment starts from a copy of the fixed-reward policy. Each episode independently samples wind strength uniformly from 0–18 and turbulence from 0–1.8, with wind enabled. This is domain randomization: vary environmental conditions during practice. Four vectorized environments feed PPO. It continues learning with the original reward only, preserving the source checkpoint. The explicit new training seed controls Python/NumPy/PyTorch and environment seeds after loading the source model.

PPO collects full rollout batches, so 200,000 requested steps produce 200,704 additional steps with this configuration. Source and additional step counts are recorded separately. The final checkpoint goes into a new `runs/robust/seed_N` folder; `config.json` records source hash, weather ranges, reward and seed. Interrupted training folders may lack a final checkpoint and are not collected as policies.

After training, both source and new checkpoint are tested for 50 episodes in each of calm, breezy and gusty weather, seed numbers 700000–700049. The results page reports original return, landing rate and Wilson intervals. This is an exploratory comparison of two saved policies. Continued learning can improve one condition and degrade another. It also changes both weather exposure and reward relative to the source's original training, so improvements cannot be attributed solely to weather randomization. Strong causal evidence requires controlled ablations, multiple training seeds and a final untouched test set. Reusing these evaluation seeds for iterative tuning turns them into development data.

The corrected seeded run (seed 2, 200,704 additional steps) produced the following 50-episode rates: source vs new policy was 90% vs 92% calm, 92% vs 88% breezy, and 68% vs 76% gusty. Gusty mean original score increased from 141.6 to 159.4, while breezy declined from 217.0 to 201.5. These are observed differences with overlapping uncertainty intervals, not proof of a general improvement. An initial implementation check revealed a seed-reset issue and was archived; the corrected run was not chosen by ranking training runs. The evaluation seed set is therefore exploratory evidence, not a pristine final benchmark.

## Files, endpoints and reproducibility

- `telemetry.py`: collection, compression, SQLite queries and episode identity.
- `predictor.py`: prefix features, grouped splits, candidate fitting, metrics and inference.
- `assistant.py`: document chunks, retrieval, exact data answers and local generation.
- `broader_training.py`: weather wrapper, fine-tuning and evaluation.
- `webapp.py`: API, local server and owned background jobs.
- `web/`: introductory page and five-tab interface.
- `artifacts/lab/`: curated telemetry, classifier, classifier report and latest broader comparison.

GET `/api/flights` lists summaries; `/api/flights/{id}` returns telemetry and forecast; `/api/summary` aggregates filters; `/api/export` exports CSV; `/api/predictor` and `/api/training` return measured reports. POST `/api/ask` handles questions. `/api/jobs` starts and inspects tasks; `/api/jobs/stop` stops the owned task. Jobs use fixed module names and validated settings, not shell commands from input. The server binds to localhost and rejects cross-origin POST requests. This is a local single-user application, not an authenticated public service.

Run `python -m pytest -q` in the project environment. New regression coverage checks telemetry round trips and idempotency, prefix-only features, split isolation, outcome exclusions, weather reproducibility, API filtering/export/input validation, and retrieval/data-answer behavior. Browser verification covers introduction, navigation, charts, replay, predictor, assistant, and training results. Saved checkpoints and classifier reports retain checksums. Dependency versions and the evaluation protocol still matter: seeded runs need not be bit-identical across platforms or library releases.

## Placement explanation and interview prompts

“I built an end-to-end local reinforcement-learning experiment and analysis application. PPO learns control from reward; a separate supervised classifier forecasts outcomes from early telemetry. SQLite stores reproducible flight histories, a dashboard compares observed reliability, and a local RAG assistant explains project evidence. I evaluated models with independent seed groups and exposed weaknesses such as low failure recall instead of reporting accuracy alone.”

Be clear that implementation was developed with AI assistance. Be prepared to explain each design decision and reproduce a small experiment yourself.

Questions to prepare: Why is PPO reinforcement learning while the failure predictor is supervised learning? What is reward hacking? Why use original rewards during evaluation? What information would cause target leakage? Why group by seed? Why does macro F1 differ from accuracy? What does a Wilson interval cover? What does a Brier score measure? What can TF-IDF miss? Why is RAG not fine-tuning? Can a cited answer still hallucinate? Why does randomized weather not guarantee robustness? What additional experiment would justify calling a model better?

Interest mapping: AI/ML covers control and classification; deep learning covers the PPO neural policy and local language model; RAG covers retrieval and grounded explanations; data analytics covers telemetry, filtering, exports, uncertainty and comparison. These components have separate responsibilities and measurable limits.
