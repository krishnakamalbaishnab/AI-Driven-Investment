# AI-Driven Investment - Cryptocurrency Price Prediction Platform

🚀 **An intelligent cryptocurrency price prediction platform powered by LSTM neural networks and real-time market data.**

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.17.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-v3.0.3-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **AI-Powered Predictions**: Advanced LSTM neural network for cryptocurrency price forecasting
- **Real-Time Data**: Fetches historical cryptocurrency data from CoinGecko API
- **Interactive Web Interface**: Modern, responsive Bootstrap-based UI
- **Data Visualization**: Beautiful Chart.js visualizations for prediction results
- **Scalable Architecture**: Modular design for easy extension and maintenance
- **Multiple Cryptocurrencies**: Support for various cryptocurrencies (currently Bitcoin)

## 🏗️ Project Structure

```
AI-Driven-Investment/
├── app.py                          # Flask web application
├── model.py                        # LSTM model training and creation
├── data_collection.py              # CoinGecko API data fetching
├── data_preprocessing.py           # Data cleaning and preparation
├── requirements.txt                # Python dependencies
├── crypto_data.csv                 # Historical price data
├── crypto_prediction_model.h5      # Trained LSTM model
├── scaler.pkl                      # Fitted data scaler
├── static/
│   └── css/
│       └── styles.css              # Custom CSS styling
├── templates/
│   ├── index.html                  # Main landing page
│   └── result.html                 # Prediction results page
└── venv/                           # Virtual environment
```

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, Flask 3.0.3
- **Machine Learning**: TensorFlow 2.17.0, Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML5, Bootstrap 5.3, Chart.js
- **API**: CoinGecko API for cryptocurrency data
- **Deployment**: Gunicorn (production ready)

## ⚡ Quick Start

### Prerequisites

- **Python 3.12 or 3.13** (Both supported)
- pip package manager
- Internet connection (for API data fetching)

**✅ Python 3.13 Supported**: This project now works with Python 3.13 using TensorFlow nightly builds.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishnakamalbaishnab/AI-Driven-Investment.git
   cd AI-Driven-Investment
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Collect training data**
   ```bash
   python data_collection.py
   ```

5. **Train the model**
   ```bash
   python model.py
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 🎯 Usage

### Training Your Own Model

1. **Collect Data**: Run `data_collection.py` to fetch historical cryptocurrency data
2. **Preprocess Data**: The `data_preprocessing.py` handles data normalization and sequence creation
3. **Train Model**: Execute `model.py` to train the LSTM neural network
4. **Make Predictions**: Use the web interface to input historical prices and get predictions

### Web Interface

1. **Home Page**: Landing page with project information and prediction form
2. **Input Data**: Enter the last 60 days of cryptocurrency prices (comma-separated)
3. **View Results**: Interactive chart displaying the predicted price

### API Integration

The platform uses the CoinGecko API to fetch real-time cryptocurrency data:
- Endpoint: `https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart`
- Default: Bitcoin historical data for 365 days
- Easily configurable for other cryptocurrencies

## 🧠 Model Architecture

The prediction model uses a sophisticated LSTM (Long Short-Term Memory) neural network:

- **Input Layer**: Sequences of 60 consecutive price points
- **LSTM Layers**: Two LSTM layers with 50 units each
- **Dropout Layers**: 20% dropout for regularization
- **Dense Layer**: Single output neuron for price prediction
- **Optimizer**: Adam optimizer with mean squared error loss

## 📊 Data Processing Pipeline

1. **Data Collection**: Fetch historical prices from CoinGecko API
2. **Normalization**: Min-Max scaling to [0,1] range
3. **Sequence Creation**: Convert data into 60-day input sequences
4. **Train/Validation Split**: 80/20 split for model training
5. **Model Training**: LSTM training with validation monitoring

## 🚀 Production Deployment

For production deployment, the application includes Gunicorn WSGI server:

```bash
gunicorn --bind 0.0.0.0:8000 app:app
```

### Environment Variables (Recommended)

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## 🔧 Configuration

- **Sequence Length**: Modify `seq_length` in `data_preprocessing.py` (default: 60 days)
- **Model Parameters**: Adjust LSTM units, dropout rates in `model.py`
- **Training Epochs**: Configure training duration in `model.py` (default: 20 epochs)
- **Cryptocurrency**: Change `crypto_id` in `data_collection.py` for different coins

## 📈 Model Performance

- **Architecture**: LSTM-based sequence prediction
- **Training**: 20 epochs with validation monitoring
- **Loss Function**: Mean Squared Error
- **Validation**: 20% holdout validation set

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest

# Format code
black .
flake8 .
```

## 📋 Future Enhancements

- [ ] Support for multiple cryptocurrencies simultaneously
- [ ] Real-time prediction updates
- [ ] Advanced technical indicators integration
- [ ] User authentication and portfolio tracking
- [ ] REST API for programmatic access
- [ ] Docker containerization
- [ ] Model performance monitoring dashboard
- [ ] Integration with additional data sources

## ⚠️ Disclaimer

**Important**: This application is for educational and research purposes only. Cryptocurrency investments carry significant financial risks. The predictions made by this model should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Krishna Kamal Baishnab**
- GitHub: [@krishnakamalbaishnab](https://github.com/krishnakamalbaishnab)

## 🙏 Acknowledgments

- [CoinGecko](https://www.coingecko.com/) for providing free cryptocurrency API
- [TensorFlow](https://tensorflow.org/) for the machine learning framework
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the responsive UI components
- [Chart.js](https://www.chartjs.org/) for data visualization

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/krishnakamalbaishnab/AI-Driven-Investment/issues) page
2. Create a new issue with detailed description
3. Contact the maintainer

---

**⭐ If you found this project helpful, please give it a star!** 