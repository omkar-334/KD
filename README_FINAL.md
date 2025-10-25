# Knowledge Distillation Comparison Experiment

This repository contains a comprehensive comparison of different Knowledge Distillation (KD) methods:

- **AT (Attention Transfer)**
- **BYOT (Bring Your Own Teacher)**
- **DML (Deep Mutual Learning)**
- **FitNets**

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements_final.txt

# Or install individually
pip install torch torchvision numpy pandas tqdm tensorboard matplotlib seaborn scikit-learn Pillow
```

### 2. Prepare Dataset

Ensure your dataset follows this structure:
```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

### 3. Run Complete Experiment

```bash
# Simple execution
python run_experiment.py

# Or run directly
python final.py
```

## 📊 What the Experiment Does

### Phase 1: Teacher Training
- Trains a **ResNet50** teacher model for 100 epochs
- Uses optimized settings for maximum performance
- Saves teacher to `checkpoints/teachers/ResNet50_teacher_best.pth`

### Phase 2: Knowledge Distillation
Runs all 4 KD methods with **ResNet18** student models:

1. **AT (Attention Transfer)**
   - Transfers attention maps from teacher to student
   - Uses temperature scaling and attention loss

2. **BYOT (Bring Your Own Teacher)**
   - Self-distillation with intermediate supervision
   - Knowledge distillation between intermediate and final outputs

3. **DML (Deep Mutual Learning)**
   - Multiple student models learn collaboratively
   - No teacher required - mutual learning between students

4. **FitNets**
   - Feature matching between teacher and student
   - Transfers intermediate feature representations

### Phase 3: Comparison & Analysis
- Compares all methods against teacher performance
- Calculates accuracy improvements
- Generates detailed comparison tables
- Saves results in multiple formats (JSON, CSV)

## 📁 Output Structure

```
logs/final_experiment_YYYYMMDD_HHMMSS/
├── final_experiment.log          # Detailed execution log
├── comparison_results.json       # Raw results data
├── comparison_results.csv        # Results table
└── experiment_summary.json       # Summary statistics

checkpoints/teachers/
├── ResNet50_teacher_best.pth     # Best teacher model
└── ResNet50_teacher_final.pth    # Final teacher model

logs/teachers/
└── ResNet50_Teacher_Final_*/     # Teacher training logs

logs/AT/
└── AT_ResNet18_Final_*/          # AT training logs

logs/BYOT/
└── BYOT_ResNet18_Final_*/        # BYOT training logs

logs/DML/
└── DML_ResNet18_Final_*/         # DML training logs

logs/FitNets/
└── FitNets_ResNet18_Final_*/     # FitNets training logs
```

## ⚙️ Configuration

### Teacher Training Settings
```python
teacher_config = {
    "model": "ResNet50",
    "epochs": 100,
    "lr": 0.1,
    "batch_size": 32,
    "step_size": 30,
    "gamma": 0.1,
}
```

### Student Training Settings
```python
student_config = {
    "model": "ResNet18",
    "epochs": 50,
    "lr": 0.01,
    "batch_size": 32,
    "temperature": 4.0,
    "alpha": 0.1,
    "beta": 1e3,
}
```

## 📈 Expected Results

The experiment will output a comparison table like this:

```
KNOWLEDGE DISTILLATION COMPARISON RESULTS
================================================================================
method    model    teacher           best_accuracy  training_time  epochs
Teacher   ResNet50 None              0.8500         1200.50        100
AT        ResNet18 ResNet50          0.8200         300.25         50
FitNets   ResNet18 ResNet50          0.8150         310.10         50
BYOT      ResNet18 ResNet50          0.8100         305.80         50
DML       ResNet18 None (Mutual)     0.8000         350.45         50
================================================================================
```

## 🔧 Customization

### Modify Experiment Parameters

Edit `final.py` to customize:

```python
# Change teacher model
teacher_config = create_teacher_config(
    model="ResNet50",  # Change to ResNet18, MultiBranchNet, etc.
    epochs=100,        # Adjust training epochs
    lr=0.1,           # Adjust learning rate
)

# Change student models
at_config = create_at_config(
    model="ResNet18",  # Change student architecture
    epochs=50,         # Adjust training epochs
    lr=0.01,          # Adjust learning rate
)
```

### Add New KD Methods

1. Create new method in respective folder
2. Import in `final.py`
3. Add to `run_knowledge_distillation_methods()` function

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in configurations
   - Use `device="cpu"` for CPU-only training

2. **Dataset Not Found**
   - Ensure `data/` directory exists
   - Check folder structure matches expected format

3. **Import Errors**
   - Install all requirements: `pip install -r requirements_final.txt`
   - Check Python path includes project root

4. **Teacher Model Not Found**
   - Teacher will be trained automatically
   - Check `checkpoints/teachers/` directory

### Debug Mode

Enable detailed logging by modifying `final.py`:

```python
logging.basicConfig(level=logging.DEBUG)  # Add this line
```

## 📚 Method Details

### AT (Attention Transfer)
- **Paper**: "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer"
- **Key Idea**: Transfer attention maps from teacher to student
- **Loss**: Classification + Attention Transfer

### BYOT (Bring Your Own Teacher)
- **Paper**: "Bring Your Own Teacher: Self-distillation with Intermediate Supervision"
- **Key Idea**: Self-distillation with intermediate supervision
- **Loss**: Classification + Knowledge Distillation + Feature Matching

### DML (Deep Mutual Learning)
- **Paper**: "Deep Mutual Learning"
- **Key Idea**: Multiple students learn collaboratively
- **Loss**: Classification + Mutual Learning

### FitNets
- **Paper**: "FitNets: Hints for Thin Deep Nets"
- **Key Idea**: Feature matching between teacher and student
- **Loss**: Classification + Feature Matching

## 📞 Support

For issues or questions:
1. Check the logs in `logs/final_experiment_*/`
2. Verify dataset structure
3. Check all dependencies are installed
4. Review configuration parameters

## 🎯 Performance Tips

1. **Use GPU**: Set `device="auto"` for automatic GPU detection
2. **Adjust Batch Size**: Larger batches for better GPU utilization
3. **Monitor Memory**: Use `nvidia-smi` to monitor GPU memory
4. **Checkpointing**: Models are saved automatically for resuming
5. **TensorBoard**: Use `tensorboard --logdir logs/` for visualization
