# AMC_Lib
AMC_Lib is an open source library for the researchers focusing on **Automatic Modulation Classification (AMC)**. In this library, we provide **standard APIs** for model training and evaluating. The code base is kept as concise and readable as possible to facilitate the development and evaluation of your own AMC models.

## 🚩 News

> 

## 🏅Leaderboard





## ✅ Usage

1. Install python and the package used in this repo. We highly recommend you to install `python==3.9`, which is the same version of the develepment, and execute the following command.
   ```python
   pip install -r requirements.txt
   ```

2. Prepare Data. You can obtain the datasets from the official websites. Add the datasets to this **path** `./Datasets/`.

   

## 🗓️ Development Plan

### * Dataset (Dataloader)

- **DeepSig Series** 
  - [ ] RadioML2016.04C
  - [ ] RadioML2016.10A
  - [x] RadioML2018.01A
  - [x] RadioML2018.01A
- HiserMod (Still waiting)
  - [ ] HiserMod

  - [ ] HiserMod

- Signal Generation (Still waiting)
  - [ ] Signal moudulate
  - [ ] Interference

  - [x] Signal modulator (⚠️Not uploaded)
  - [ ]  Interference


### * Methods

- Traditional Methods
  - [ ] SVM
  - [ ] Decision Tree
- Baselines
  - [ ] LSTM, GRU, CNN, ResNet, DenseNet
  - [ ] CLDNN
  - [ ] TCN
  - [ ] Transformer
  - [ ] Mamba
  - [ ] KAN
  - [ ] miniGRU, miniLSTM
- Time series classification methods (Thanks to [TSLib](https://github.com/thuml/Time-Series-Library))
  - [ ] (Not yet!!)

### * Evaluation Tools

- Complexity
  - [ ] Parameters
  - [ ] Memory
  - [ ] FLOPs
  - [ ] Inference Speed
- Efficiency
  - [x] Accuracy

### * Preprocessing



## 😃Citation

If you find this repo useful, please cite our paper.



## 📨Contact

If you have any questions or suggestions, feel free to contact us:

Email: yixuan.dyf@gmail.com

## 😘Thanks to

- Time Series Library: https://github.com/thuml/Time-Series-Library

