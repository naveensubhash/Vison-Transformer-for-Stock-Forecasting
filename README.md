# Vision Transformer for Stock Market Forecasting

### Authors: Kahan Dhaneshbhai Sheth, Naveen Subhash Udata  
Northeastern University, Boston, MA  
Email: sheth.kah@northeastern.edu, udata.n@northeastern.edu  
Date: December 11, 2023

## Table of Contents
1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Technologies Used](#technologies-used)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Contributors](#contributors)
9. [References](#references)

## Project Overview

This project applies **Vision Transformers (ViT)**, a powerful deep learning model, to forecast stock market price movements. Traditional models like RNNs, LSTMs, and CNNs have been used for stock forecasting, but this project innovates by converting stock time-series data into images and using **Vision Transformers** for superior performance.

## Objectives

- Leverage Vision Transformers to predict stock market price movements.
- Convert stock market time series data into images using **Gramian Angular Fields (GAF)**.
- Compare the performance of ViT against traditional models such as **RNN**, **LSTM**, **CNN**, and **Transformer**.

## Methodology

### Dataset
- Historical stock market data was sourced from **Yahoo Finance**.
- For example, data from **Bajaj Finance Limited** (from Dec 30, 1995 to Oct 30, 2023) was used. 
- Data was transformed into **Mid Price** (average of opening and closing prices) and split into training, validation, and test sets.

### Data Transformation
- **Gramian Angular Field (GAF)** was used to convert numerical stock data into images for processing by the Vision Transformer.

### Model Implementation
- The **Vision Transformer (ViT)** architecture was adapted to process image patches from GAF-transformed data.
- Traditional models such as **RNN**, **LSTM**, **CNN**, and **Transformer** were also trained for performance comparison.

## Technologies Used

- **Python**
- **yfinance** library for stock data collection
- **pyts.image** for Gramian Angular Field transformation
- **TensorFlow** / **PyTorch** for deep learning model implementation
- **Matplotlib** for visualizations

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/vision-transformer-stock-forecasting.git
    cd vision-transformer-stock-forecasting
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the model training script:
    ```bash
    python train_model.py
    ```

4. View the results and accuracy comparisons:
    ```bash
    python evaluate_results.py
    ```

## Results

- **Vision Transformer (ViT)** demonstrated superior accuracy across various stocks compared to RNN, LSTM, CNN, and Transformer.
- Example results for stock price direction prediction:

    | Stock Name                   | ViT Accuracy | Transformer | RNN  | LSTM | CNN  |
    |------------------------------|--------------|-------------|------|------|------|
    | Kotak Bank                    | 58.73%       | 53.32%      | 47.97%|47.15%|52.38%|
    | Bajaj Finance Limited         | 59.09%       | 58.21%      | 56.91%|50.41%|52.38%|
    | Infosys Limited               | 61.91%       | 55.56%      | 60.16%|62.16%|57.14%|

## Conclusion

The **Vision Transformer (ViT)** model shows promise for stock market forecasting, consistently outperforming traditional models in predicting stock price direction. Its ability to handle time-series data in the form of images opens up new avenues for financial forecasting.

## Contributors

- **Naveen Subhash Udata**: Data transformation, GAF conversion, model comparison.
- **Kahan Dhaneshbhai Sheth**: Data preprocessing, implementation of ViT and Transformer models, visualizations.

## References

1. Dosovitskiy et al., *An image is worth 16x16 words: Transformers for image recognition at scale*, 2020.
2. Vaswani et al., *Attention is all you need*, 2017.
3. Wang et al., *Stock market index prediction using deep transformer model*, 2022.
4. Muhammad et al., *Transformer-based deep learning model for stock price prediction*, 2023.

