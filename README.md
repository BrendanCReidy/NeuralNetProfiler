# Instructions

1. Clone repository
```
git clone https://github.com/BrendanCReidy/NeuralNetProfiler.git
```

2. Set flag to the number of layers you want to profile
```
flagLayer = 1 # Profiles only the first layer
```
```
flagLayer = 5 # Profiles the first five layers
```

The only line of code responsible for inference is line 33
```
intermediate_model.predict(features) # This is the only line responsible for inference
```

## Important Note
The profiling is cumulative. If the flagLayer is set to 5 you will be profiling layers 1-5
