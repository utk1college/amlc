# Running Stage 1 v3 on the WSL box

Run every block in order. Commands after step 1 run **inside WSL**.
Long steps run inside `tmux`: detach with `Ctrl-b d`, reattach with `tmux attach -t amlc`.

## 1. Open WSL (Windows PowerShell)

```powershell
wsl
```

## 2. Check resources (paste the output back)

```bash
nproc; free -g; df -h ~; nvidia-smi || true
```

If WSL shows much less RAM or fewer CPUs than the machine has, do this in **PowerShell**, then reopen WSL:

```powershell
notepad $env:USERPROFILE\.wslconfig
```

Put in the file (adjust the numbers to the machine):

```
[wsl2]
memory=56GB
processors=16
```

```powershell
wsl --shutdown
```

## 3. Tools and folder (Linux filesystem, never /mnt/c or /mnt/e)

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-dev build-essential git tmux unzip wget
```

```bash
mkdir -p ~/amlc_work ~/models && cd ~
```

## 4. Clone the repo and download the data

```bash
git clone https://github.com/utk1college/amlc.git ~/amlc
```

```bash
wget -O ~/student_resource.zip https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip && unzip -q ~/student_resource.zip -d ~/amlc_data
```

```bash
find ~/amlc_data -type d -name dataset
```

Use the path printed above (it ends in `student_resource/dataset`) in step 5.

## 5. Python environment and variables

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate && pip install -U pip
```

```bash
pip install -r ~/amlc/code/business_entity_resolution/requirements.txt
```

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && pip install transformers==5.17.0 faiss-cpu==1.15.1 huggingface_hub
```

```bash
hf download intfloat/multilingual-e5-small --local-dir ~/models/multilingual-e5-small
```

Set these in **every new shell**. Replace the DATA_DIR path with the one from step 4:

```bash
source ~/venv/bin/activate
export DATA_DIR=~/amlc_data/student_resource/dataset
export WORK_DIR=~/amlc_work OUTPUT_DIR=~/amlc_output E5_DIR=~/models/multilingual-e5-small PYTHON=python3
```

## 6. Unit tests (should print OK)

```bash
cd ~/amlc && python -m unittest discover -s code/business_entity_resolution/tests
```

## 7. v2 baseline on the old code (about 1.5 h)

This gives the v2 candidates, the 0.9696 baseline and the box's throughput.

```bash
tmux new -s amlc
```

```bash
git -C ~/amlc worktree add ~/amlc_v2 2c9b92f
```

```bash
cd ~/amlc_v2/code/business_entity_resolution && ./run_pipeline.sh all 2>&1 | tee ~/amlc_work/baseline.log
```

Paste back the last 60 lines of `~/amlc_work/baseline.log`.

## 8. Phase 0 diagnostics on the baseline

```bash
cd ~/amlc/code/business_entity_resolution/src
```

```bash
python diagnostics.py loss --val-scores $WORK_DIR/model/val_scores.parquet --truth $WORK_DIR/split/val_ground_truth.tsv --data-dir $DATA_DIR --out $WORK_DIR/diag_loss.json
```

```bash
python diagnostics.py shift --val-scores $WORK_DIR/model/val_scores.parquet --test-scores $WORK_DIR/model/test_scores.parquet --truth $WORK_DIR/split/val_ground_truth.tsv --data-dir $DATA_DIR --out $WORK_DIR/diag_shift.json
```

```bash
python robustness.py adversarial --features v2 --val-features-dir $WORK_DIR/features/train --test-features-dir $WORK_DIR/features/test --val-source1 $DATA_DIR/train/train_source1.tsv --test-source1 $DATA_DIR/test/test_source1.tsv --out $WORK_DIR/diag_adv.json
```

## 9. Blocking lab (new code)

```bash
cd ~/amlc/code/business_entity_resolution
```

```bash
./run_lab.sh dump 2>&1 | tee $WORK_DIR/lab_dump.log
```

Note the chosen `"w"` from `$WORK_DIR/lab/choose_w.json`, then put it in `W`:

```bash
W=$(python -c "import json; print(json.load(open('$WORK_DIR/lab/choose_w.json'))['w'])"); echo $W
```

```bash
./run_lab.sh reverse $W 2>&1 | tee $WORK_DIR/lab_reverse.log
```

```bash
./run_lab.sh embed train name 2>&1 | tee $WORK_DIR/lab_embed_name.log
```

```bash
./run_lab.sh embed train name_address 2>&1 | tee $WORK_DIR/lab_embed_name_address.log
```

**Stop here and paste back:** `choose_w.json`, the reverse and embed reports, and the step 2 output. I'll send the budget `B` for the next command.

```bash
./run_lab.sh select <B> $W
```

```bash
./run_lab.sh confirm && ./run_lab.sh oracle && ./run_lab.sh export && ./run_lab.sh gate2
```

Paste back `select.json`, `confirm.json`, `oracle.json` and `gate2.json` from `$WORK_DIR/lab/`.

## 10. Full v3 pipeline (only if Gates 1 and 2 pass)

```bash
./run_pipeline.sh all 2>&1 | tee $WORK_DIR/v3.log
```

Paste back `$WORK_DIR/model3/report.json` and the last 40 lines of `$WORK_DIR/v3.log`.
The submission is `$OUTPUT_DIR/matching_results.tsv`, with `$OUTPUT_DIR/candidate_pairs.tsv` beside it.
