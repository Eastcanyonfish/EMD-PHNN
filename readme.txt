##PHNN Model for Candlestick Technical Trading Strategy

#Background

The PHNN (Parallel Hybrid Neural Networks) model is designed to enhance return prediction using a candlestick technical trading strategy. It incorporates two sub-networks with configurable CNN or LSTM architectures. The final prediction is generated through a fully connected neural layer that combines outputs from these sub-networks. The model is trained on prices' components decomposed through Empirical Mode Decomposition for improved accuracy in trend and candlestick pattern recognition.
Files Description

    - EMD.py: Code for decomposing prices into high and low components using Empirical Mode Decomposition.
    - datapreproc.py: Code for reshaping components to meet input requirements.
    - PHNNmodel: Establishes sub-network architectures for four combinations: "CNN-LSTM", "LSTM-LSTM", "LSTM-CNN", "CNN-CNN".
    - Evaluation.py: Metrics for assessing model performance.
    - Demo.py: Example implementation of the EMD-PHNN model.

#Data Description

The data used pertains to the Chinese A shares composite index, including High, Low, Open, and Close prices.

#Usage

To utilize the PHNN model:
  Demo.py provides an example implementation. The example showcases how to decompose prices and then reshape components. Through Choosing a sub-network architecture for PHNNmodel, we can train the PHNN model on the prices components.  

  

#Conclusion

The PHNN model offers a promising approach to enhancing return prediction in candlestick technical trading strategies. It outperforms traditional benchmark models by leveraging parallel hybrid neural networks and specialized sub-networks.

#Citation

Min Zhu, Yu Guo, Yuping Song, "A parallel hybrid neural networks model for forecasting returns with candlestick technical trading strategy," Expert Systems with Applications, Volume 255, Part A, 2024, 124486, ISSN 0957-4174, DOI Link

Keywords: Parallel hybrid neural networks, Empirical mode decomposition, Candlestick trading technique, Returns prediction
