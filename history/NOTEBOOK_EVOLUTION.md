# Classroom Behavior Detection – Notebook Evolution

Goal: Summarize changes and reasoning across the notebooks `history/behavior-detection (1)–(11)`, which train models to classify student behaviors in a classroom.

## Version-by-Version Changes
- **(1)**  
  - What: Baseline. `/kaggle/input/classroom-student-behaviors/Behaviors_Features`, EfficientNet-B0 single-image classifier, GroupShuffleSplit by sequence, light aug, 1 epoch, saves `best_model.pth` / `label_map.json`.  
  - Why: Quickly validate the pipeline and get an initial accuracy benchmark with a minimal setup.

- **(2)**  
  - What: Same data; switched to ConvNeXt-Small (in22ft1k), label smoothing, AMP, OneCycleLR for 10 epochs; inference cells commented out.  
  - Why: Stronger backbone and regularization to lift accuracy while keeping training efficient via AMP.

- **(3)**  
  - What: Switched to merged data `Behaviors_Features_Final`; otherwise matches (2).  
  - Why: Improve generalization with more/cleaner data.

- **(4)**  
  - What: Moved to sequence classification. Build 8-frame clips per group; WeightedRandomSampler + class weights for imbalance; introduced TemporalMeanNet (ConvNeXt features averaged over time).  
  - Why: Actions depend on temporal context; sequence aggregation should outperform single-frame, while addressing class imbalance.

- **(5)**  
  - What: Split by person to prevent ID leakage; stronger aug (ColorJitter/Blur/Erasing etc.) and random continuous clip sampling; Mixup/CutMix + SoftTargetCrossEntropy; TemporalMeanNet (ConvNeXt fb_in22k); early stopping.  
  - Why: Person-level split yields more realistic generalization. Strong aug + Mixup/CutMix for robustness; early stopping to curb overfitting.

- **(6)**  
  - What: Removed class `Standing`, reducing to 6 classes; otherwise same as (5).  
  - Why: Simplify the task and improve balance/quality.

- **(7)**  
  - What: Added local CutMix targeting Writing/Reading (hand region), combined with Mixup; model remains TemporalMeanNet.  
  - Why: Emphasize hand cues to reduce confusion between Reading/Writing.

- **(8)**  
  - What: Added “deploy-like” validation (blur/crop/color jitter) and prioritized deploy-val for early stopping (tie-break with clean-val); TemporalMeanNet + local CutMix/Mixup.  
  - Why: Anticipate camera/domain shifts and select models that stay stable in real settings.

- **(9)**  
  - What: Switched to TemporalConvNet (1D conv over time). Kept dual validation (clean/deploy) and local CutMix/Mixup. Saved as `update_convnext.pth`.  
  - Why: Capture temporal structure more richly than simple averaging to better model motion patterns.

- **(10)**  
  - What: Still TemporalConvNet. Upweighted Reading/Writing in sampler and loss (Sleeping down-weighted); DataParallel support; checkpoint `strengthen_writing_reading.pth`.  
  - Why: Push recall on hard classes by shifting training distribution; multi-GPU for stability/speed.

- **(11)**  
  - What: Tweaked class weights again (Reading/Writing still boosted; Sleeping/Turning_Around down; Looking_Forward slightly down). Same pipeline as (10). Checkpoint `weaken_turning_around.pth`.  
  - Why: Fine-tune misclassification costs to retain main classes while reducing specific errors.

## Overall Trajectory
- **Data**: Original → merged (v3–) → drop Standing (v6–).  
  - Reason: Better data quantity/quality; simplify the label space.
- **Splits**: Sequence-based → person-based (v5–).  
  - Reason: Prevent ID leakage; evaluate closer to real-world deployment.
- **Models**: Single-image EfficientNet → ConvNeXt → TemporalMeanNet (time avg) → TemporalConvNet (temporal conv).  
  - Reason: Gradually leverage stronger backbones and temporal structure for action recognition.
- **Regularization/Robustness**: Stronger aug, Mixup/CutMix → local CutMix (hand emphasis) → deploy-like validation.  
  - Reason: Reduce overfitting, highlight key regions, and ensure robustness to camera/domain shifts.
- **Imbalance Handling**: Progressive class/sampling weights prioritizing Reading/Writing.  
  - Reason: Improve recall on target classes by rebalancing learning signals.***
