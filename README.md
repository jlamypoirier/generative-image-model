# generative-image-model


This is the code for a neural network generation and training framework, nnBuilder. nnBuilder aims to be a lightweigth, flexible and extensible framework to generate and run TensorFlow graphs from a JSON-like definition. The goal is to make experimentation easy by allowing to quickly change the network architecture, without any restriction on what the network looks like.

In this framework, all graph are made from basic building blocks, or "layers", which can be combined into other layers in a tree-like fashion. They are built through a layer factory, which can build arbitrarily complex graphs from JSON-like code (nested lists and dictionnaries). The framework automatically handles many tasks which are otherwise tedious in TensorFlow with minimal work, for example:

-Copying graphs, with or without parameter sharing (essential for running the graph on a different input)

-Managing sessions, and initializing the graph

-Saving and loading variable definitions for a specific network

-Setting up and running the graph trainer, and feeding the training parameters

-Running the test phase

-Managing the training data: loading , batch generation, labels, preprocessing, input pipeline (in progress)

See the code and examples for more details.

The framework evolved from the code for a project on generative models of images, which is now paused until the nnBuilder framework is powerful enough for the project.
