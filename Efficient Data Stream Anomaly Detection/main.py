import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import norm



class AnomalyDetector:
    """
    A class for detecting anomalies in a continuous data stream using z-score method.
    
    This detector uses a sliding window approach to adapt to concept drift and seasonal variations.
    It calculates z-scores based on the recent data points within the window and flags anomalies
    when the z-score exceeds a specified threshold.
    
    Attributes:
        window_size (int): The number of recent data points to consider for z-score calculation.
        threshold (float): The z-score threshold above which a point is considered an anomaly.
        data_window (deque): A sliding window of recent data points.
        anomalies (list): A list to store detected anomalies.
    """

    def __init__(self, window_size=100, threshold=3):
        """
        Initialize the AnomalyDetector.

        Args:
            window_size (int): Size of the sliding window. Default is 100.
            threshold (float): Z-score threshold for anomaly detection. Default is 3.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.data_window = deque(maxlen=window_size)
        self.anomalies = []

    def detect(self, value):
        """
        Detect if a given value is an anomaly.

        This method adds the new value to the sliding window and calculates its z-score
        based on the current window's statistics. If the absolute z-score exceeds the
        threshold, the value is flagged as an anomaly.

        Args:
            value (float): The data point to be checked for anomaly.

        Returns:
            bool: True if the value is an anomaly, False otherwise.
        """
        self.data_window.append(value)
        
        if len(self.data_window) < self.window_size:
            return False
        
        mean = np.mean(self.data_window)
        std = np.std(self.data_window)
        
        # Avoid division by zero
        if std == 0:
            return False
        
        z_score = (value - mean) / std
        
        is_anomaly = abs(z_score) > self.threshold
        if is_anomaly:
            self.anomalies.append(value)
        
        return is_anomaly

def generate_data_stream(n_points, seasonality=24, trend=0.1, noise_level=0.5):
    """
    Generate a simulated data stream with seasonal patterns, trend, and noise.

    Args:
        n_points (int): Number of data points to generate.
        seasonality (int): Period of the seasonal pattern. Default is 24.
        trend (float): Slope of the linear trend. Default is 0.1.
        noise_level (float): Standard deviation of the random noise. Default is 0.5.

    Returns:
        numpy.ndarray: Array of generated data points.
    """
    t = np.arange(n_points)
    seasonal = np.sin(2 * np.pi * t / seasonality)
    trend = trend * t
    noise = np.random.normal(0, noise_level, n_points)
    return seasonal + trend + noise

def visualize_stream(data, anomalies, window_size=100):
    """
    Visualize the data stream and detected anomalies.

    Args:
        data (list or numpy.ndarray): The data stream.
        anomalies (list): Indices of detected anomalies.
        window_size (int): Size of the sliding window for visualization. Default is 100.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data Stream')
    plt.scatter(anomalies, [data[i] for i in anomalies], color='red', label='Anomalies')
    plt.title('Data Stream with Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def main():
    """
    Main function to run the anomaly detection on a simulated data stream.
    """
    n_points = 1000
    data_stream = generate_data_stream(n_points)
    
    # Inject some anomalies
    anomaly_indices = [100, 250, 500, 750]
    for idx in anomaly_indices:
        data_stream[idx] += np.random.uniform(5, 10)
    
    detector = AnomalyDetector()
    detected_anomalies = []
    
    for i, value in enumerate(data_stream):
        try:
            if detector.detect(value):
                detected_anomalies.append(i)
        except Exception as e:
            print(f"Error processing data point {i}: {e}")
    
    visualize_stream(data_stream, detected_anomalies)

if __name__ == "__main__":
    main()

# Algorithm Explanation:
"""
This script implements a simple yet effective anomaly detection algorithm based on z-scores.
The z-score method is chosen for its simplicity, adaptability, and effectiveness in detecting
outliers in a continuous data stream.

Key features of the algorithm:
1. Sliding Window: The algorithm maintains a sliding window of recent data points, allowing
   it to adapt to concept drift and seasonal variations in the data.

2. Z-score Calculation: For each new data point, the algorithm calculates its z-score based
   on the mean and standard deviation of the current window. This standardizes the data and
   makes the method robust to scale changes.

3. Threshold-based Detection: A data point is flagged as an anomaly if its absolute z-score
   exceeds a predefined threshold. This allows for easy tuning of the algorithm's sensitivity.

4. Efficiency: The use of a deque for the sliding window ensures constant-time append and pop
   operations, making the algorithm efficient for real-time processing.

Effectiveness:
- The algorithm is effective in detecting sudden spikes or drops in the data stream.
- It can adapt to gradual changes in the data distribution due to the sliding window approach.
- The method is computationally efficient, making it suitable for real-time applications.
- It's easy to understand and implement, which facilitates maintenance and modifications.

Limitations:
- The algorithm might not be as effective for detecting more complex anomaly patterns.
- It assumes a roughly normal distribution of data within the window, which may not always hold.
- The fixed threshold might not be optimal for all types of data streams.

Despite these limitations, this algorithm provides a solid foundation for anomaly detection
in many real-world scenarios and can be easily extended or refined as needed.
"""