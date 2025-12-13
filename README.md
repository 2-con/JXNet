<p align="center" style="font-size:70px">
  <img width="1047" height="226" alt="Screenshot 2025-08-14 211650" src="https://github.com/user-attachments/assets/e4388a71-a6ef-4f0a-be2c-58796424c64f" />
  <b>JXNet</b>
</p>

# **JXNet** 🧠

### **A free and open framework for Machine Learning**

---

### What is JXNet ❓
JXNet is the direct continuation of PyNet. A free and open-source software framework for machine learning and artificial intelligence. Unlike PyNet, JXNet uses JAX and XLA for performance speedup which allows the framework to focus on medium to large projects / experiments by providing a highly modular API and debugging infrastructure for model evaluation. JXNet is meant for full custoization which means the requirements for a custom layer is a single inheretance call from a base layer.

JXNet is committed to democratizing and opening up machine learning and artificial intelligence to the world, so from the newest student to the most experienced scientist, we are determined to share a new technological revolution that is on the horizon.

---

### Getting Started 🚀

JXNet's interface are designed to be intuitive and modular, simmilar to PyNet.

#### JXNet StandardNet API
The StandardNet API is the main framework to build models. Most processes and intuition remain consistent across all other interfaces within JXNet with slight differences in systems or operational execution.

    # create a model instance
    MODEL = sequential(
      layer1,
      layer2,
      ...
    )

    # JXNet support implicit args by passing strings instead of argument instances
    MODEL.add(layer1(...))
    MODEL.add(layer2(...))
    ...

    # model must be compiled
    MODEL.compile(
      ...
    )

    # train the model. make sure the inputs are of the appropriate datatype
    MODEL.fit(feature,labels)

    # the model can be used after training.
    MODEL.push(input)
  
#### JXNet NetLab API
The JXNet NetLab API is the main framework to diagnose and process sets of StandardNet models in bulk. Its goal is to streamline model testing and experimentation thorugh automatic iteration and logging.

    # pass in StandardNet models in order of testing
    ENVIRONMENT = Sample(
      model1,
      model2,
      ...
    )

    # define the procedure to apply per model
    ENVIRONMENT.procedure(...)

    # experiment settings
    ENVIRONMENT.compile(...)

    # run the experiment on datasets
    ENVIRONMENT.run(dataset)

#### Other models
JXNet also have other models for regression, classification and clustering under the "models" Module.

    # using a linear regression as an example
    MODEL = Linear(...)

    # similar to StandardNet
    MODEL.compile(...)

    # only some models have a "fit" method, some like K-Nearest Neighbors dont have this method at all.
    MODEL.fit(features, labels)

    # simmilar to the "push" method from the core APIs
    MODEL.predict(...)

---

### Core Features ⚡
At its core, JXNet is a framework full of layers and processes that require complex setup to run properly, hence, prebuilt APIs are made in order to streamline this process. Currently, JXNet hosts four main APIs that could be used to abstract processes and make development easier and faster.

---

### Active JXNet APIs
---
#### StandardNet API
A high-performance API built around the JAX ecosystem, leveraging JNP operations and JIT-compiled systems to boost calculations up to 5x the speed thanks to the XLA compiler. Everything is designed to be modular, so a custom layer can be passed into the sequential class as long as it adheres to NetFlash specifications.

**Built-in Learnable layers:**
- Multiheaded Self-Attention
- Fully Connected
- Locally Connected
- Multichannel Convolution
- Multichannel Deconvolution
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Recurrent (Simple Recurrent)
- Normalization
  - Batchnorm
  - Layernorm
  - Instancenorm

**Built-in Functional layers:**
- Operation (Multipurpose)
- Multichannel MaxPooling
- Multichannel MeanPooling
- Flatten
- Dropout
- Reshape

---
#### NetLab API
An experiment-oriented API using JNP and the JAX ecosystem, unlike NetFlash, NetLab aims to provide an easier way to experiment and configure tests by providing additional features and scaling back some features. NetLab encourage the use of custom implimentations as long as it follows strict guidelines on how to design such implimentations.

**Built-in Procedures:**
- Compile
- Train
- Evaluate
- Values
- Track Layers

---

### Additional Features ⚒️
Aside from APIs and layers, JXNet also contains other features that could aid in your project or experiment.  

#### Arraytools
Tools for dealing with arrays and lists in python. While not as extensive as NumPy or JNP, it is still quite useful for some non-vectorized models.

#### Utility
Functions for time-related visualization and measurement.

#### Visual
Functions used to display and visualize Python datatypes and JXNet objects, useful for debugging.

---
### Regressors

JXNet's regression models provide a diverse set of tools for predicting continuous values. These models are self-contained and easy to use, making them ideal for understanding the fundamentals of curve fitting and trend analysis.

- Linear
- Polynomial
- Logistic
- Power
- Exponential
- Logarithmic

---
### Classifiers

The classification suite offers a range of models for predicting discrete categories. These algorithms are perfect for learning about different approaches to pattern recognition and decision-making.

- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Naive Bayes
- SVM (Support Vector Machines)
- MSVM (Multiclass Support Vector Machines)

---

### Installation 📲
Despite self-sufficiency as a core philosophy, JXNet still needs some modules for core operations and processing.

**Dependencies**
- JAXLIB
- JAX
- Time
- Matplotlib

---

### License ⚖️
This project is licensed under the Apache License 2.0 (January 2004) for distribution and modification.

**[Apache License 2.0 (January 2004)](https://github.com/2-con/JXNet/blob/main/LICENSE)**

---

### Contributors 🤝
JXNet is a project that branched off PyNet in October 2025 and has been receiving updates ever since from one person, any help is much appreciated.

**Maintainer**
2-Con
