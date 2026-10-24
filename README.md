# Kithul Flower Classifier

Image classifier that identifies Kithul, Coconut, and Date palm flowers (Kithul palm project).

## Uncertainty Rejection & Calibration

Because the deep-learning model is trained on a closed set of 3 classes, out-of-distribution images (non-palm flowers, leaves, objects) could otherwise produce high softmax scores due to softmax normalization.

To prevent false positive classifications, the backend implements an **uncertainty rejection gate** based on:
1. **Raw Max Logit (`MAX_LOGIT_MIN`)**: The model's unnormalized activation for the top predicted class.
2. **Top-2 Margin (`MARGIN_MIN`)**: Difference between the highest and second-highest class probabilities ($P_{\text{top1}} - P_{\text{top2}}$).

If `max_logit < MAX_LOGIT_MIN` or `margin < MARGIN_MIN`, the API returns `"status": "uncertain"`, and the frontend displays `"Not recognised as a Kithul, Coconut or Date palm flower"`.

### Running Threshold Calibration

To calibrate `MAX_LOGIT_MIN` and `MARGIN_MIN` from data:

```powershell
python ml/calibrate_thresholds.py --real data/test --junk junk --out backend/thresholds.json
```

* **Inputs**:
  * `--real`: Directory containing real palm test images (`data/test/<class>/*.jpg`).
  * `--junk`: Directory containing non-palm lookalike images (`junk/`).
* **Output**: Writes the calibrated values to `backend/thresholds.json`.

> [!IMPORTANT]
> **Re-calibration Required After Retraining:**
> Every time the model is retrained (`train_flower_classifier.py`), the model's weights and logit scales change. You **must re-run the calibration script** after any retraining run so that `backend/thresholds.json` reflects the new logit distribution.

---

## Running the Application

1. **Calibrate Thresholds (if not already done):**
   ```powershell
   python ml/calibrate_thresholds.py
   ```

2. **Start Backend Server:**
   ```powershell
   python -m uvicorn backend.main:app --port 8000 --reload
   ```

3. **Open the Demo UI:**
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.
