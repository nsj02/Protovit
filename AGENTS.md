# Repository Guidelines

## Repository Orientation
- Primary training logic lives in `main.py`, `train_and_test.py`, `model.py`, and helper utilities (`helpers.py`, `save.py`, `log.py`).
- Data utilities (`preprocess.py`, `img_aug.py`, `tools/`) prepare CUB/Stanford Cars datasets into `datasets/…` directories consumed by the trainers.
- Interpretation scripts (`local_analysis.py`, `global_analysis.py`, `find_nearest.py`, `analysis_settings.py`) surface prototype reasoning, while `spatial_alignment_test/` hosts robustness experiments.

## Code Reading & Annotation Workflow
1. **Follow the data -> settings -> training -> analysis order.** Start with preprocessing scripts to grasp directory conventions, then move through configuration files, training loops, and finally interpretability/robustness modules.
2. **Before reading, skim `progress_log.md`** to locate remaining unchecked items; mark your target file.
3. **As you read, add concise Korean comments** explaining tensor shapes, control flow, and non-obvious hyperparameters. Prefer block comments above functions or logic chunks rather than inline noise. Never rename variables or alter behaviour while annotating.
4. **After finishing a file**, update its checklist entry in `progress_log.md` with a function-by-function bullet list (이름, 입력/출력, 핵심 역할) so 협업자가 한눈에 흐름을 파악할 수 있게 한다. 진행 노트 표에는 미해결 이슈와 날짜를 함께 남긴다.

## Commenting Standards
- Keep comments actionable: describe “why” and expected inputs/outputs, not line-by-line rephrasings.
- Mention assumptions (e.g., requires 3-channel tensor, expects pre-cropped images) so downstream readers know prerequisites.
- When documenting loops over GPUs or prototypes, highlight complexity costs or potential bottlenecks.
- Use Korean for explanations, but retain symbol names (`prototype_vectors`, `gpuid`) to avoid ambiguity.

## Progress Tracking
- `prompt/progress_log.md` is the single source of truth. Do not duplicate notes elsewhere.
- Checklist status: `[ ]` 미검토, `[x]` 주석 및 이해 완료. If partial, leave unchecked and describe what remains.
- 진행 노트 표는 최소 날짜, 파일, 요약, 후속 작업 4열을 모두 채웁니다. TODO가 해결되면 “완료”로 남겨 재검증 필요성을 없앱니다.

## Collaboration & Commits
- Batch related annotations into clear commits (e.g., `Annotate preprocess helpers`, `Document training loop control flow`).
- Commit messages use present-tense imperatives under ~72 characters; include Korean 설명은 선택.
- Push regularly so teammates can sync updated comments and log progress without conflicts.
