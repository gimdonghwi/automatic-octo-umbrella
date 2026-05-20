# DAIC-WOZ EDA Setup Guide

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Data Paths

**Option A: Environment Variable (Recommended)**
```bash
export DAIC_ROOT=/path/to/your/daic/data
```

**Option B: Edit config/paths.yaml**
```yaml
data:
  root_dir: /your/absolute/path/to/daic/data
```

### 3. Verify Installation

```bash
python -c "from eda import DAICDataLoader; print('✓ Installation successful')"
```

### 4. Run Analysis

**Interactive Notebooks:**
```bash
jupyter notebook notebooks/01_data_profiling.ipynb
```

**Programmatic Execution:**
```python
from eda import DAICDataLoader, ComprehensiveFeatureProfiler

loader = DAICDataLoader.from_config('config/paths.yaml')
profiler = ComprehensiveFeatureProfiler(loader)
features_df = profiler.profile_dataset('train')
print(features_df.head())
```

---

## Expected Data Structure

Your DAIC data directory should contain:

```
$DAIC_ROOT/
├── train_split_Depression_AVEC2017.csv
├── dev_split_Depression_AVEC2017.csv
├── full_test_split.csv
├── DAIC-WOZ_transcription_data.csv
├── audio_embeddings/
│   ├── 300_embeddings.pt
│   ├── 301_embeddings.pt
│   └── ...
└── text_embeddings/
    ├── 300_embeddings.pt
    ├── 301_embeddings.pt
    └── ...
```

---

## Troubleshooting

### "FileNotFoundError: Data root not found"
- Check `DAIC_ROOT` environment variable: `echo $DAIC_ROOT`
- Verify path in `config/paths.yaml`
- Ensure data directory exists and is readable

### "ModuleNotFoundError: No module named 'eda'"
- Ensure you're running from project root: `cd daic_woz_analysis`
- Check virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### "Missing audio/text embeddings"
- Run `loader.validate_data_availability('train')` to check
- Some participants may legitimately lack certain modalities
- System will log warnings but continue processing

---

## Next Steps

1. ✅ Run `01_data_profiling.ipynb` - Verify data quality
2. ✅ Run `02_depression_analysis.ipynb` - Extract insights
3. ✅ Run `03_question_type_analysis.ipynb` - Inform model design
4. 📊 Review generated reports in `reports/`
5. 🔬 Proceed to model development with informed features

---

**Questions?** Check README.md or open an issue on GitHub.
