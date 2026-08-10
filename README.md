# 📊 Financial Anomaly Interpretability Using LLMs (F.A.I.L)

### MA5750: Object Oriented Programming Course Project by Aritra Dasgupta (MA25M005), Ashwini (MA25M006), Devharish N (MA25M010), Rumaiz Ibrahim K (MA25M023).

### Now hosted on https://failoops-jp6apprngwv9m5atkmlf4qh.streamlit.app !!

A sophisticated quantitative analysis framework implementing Object-Oriented Programming principles, SOLID principles, and design patterns for comprehensive market analysis.
 
> **📚 [OOP Concepts Documentation](./OOP_CONCEPTS.md)** - Comprehensive guide to all OOP concepts, design patterns, and SOLID principles used in this project.

## 🏗️ Architecture Overview

This project demonstrates advanced OOP concepts including:

### 🎯 Design Patterns Implemented
- **Strategy Pattern**: Different algorithms for data loading, anomaly detection, news retrieval, embedding generation, similarity analysis, and AI explanations
- **Factory Pattern**: Creating different types of processors and pipelines
- **Observer Pattern**: Progress tracking and UI updates
- **Template Method Pattern**: Pipeline execution flow
- **Builder Pattern**: Complex pipeline configuration
- **Repository Pattern**: Data persistence and retrieval

### 🔧 SOLID Principles
- **Single Responsibility**: Each class has one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Derived classes are substitutable for base classes
- **Interface Segregation**: Small, focused interfaces
- **Dependency Inversion**: Depend on abstractions, not concretions

## 📁 Project Structure

```
quant_analysis/
├── core/                           # Core framework (Abstractions & Base Classes)
│   ├── __init__.py
│   ├── base.py                     # Base classes and common functionality
│   ├── interfaces.py               # Abstract interfaces and contracts
│   └── exceptions.py               # Custom exception hierarchy
├── processors/                     # Data processing components (Strategy Pattern)
│   ├── __init__.py
│   ├── data_loader.py              # Data loading strategies
│   ├── anomaly_detector.py         # Anomaly detection strategies
│   ├── news_retriever.py           # News retrieval strategies
│   ├── embedding_generator.py      # Embedding generation strategies
│   ├── similarity_analyzer.py      # Similarity analysis strategies
│   └── ai_explainer.py            # AI explanation strategies
├── pipeline/                       # Pipeline orchestration (Template Method)
│   ├── __init__.py
│   ├── quantitative_pipeline.py    # Main pipeline (Template Method)
│   ├── progress_observer.py        # Progress tracking (Observer Pattern)
│   └── pipeline_factory.py        # Pipeline creation (Factory Pattern)
├── ui/                            # User interface components (MVC Pattern)
│   ├── __init__.py
│   ├── streamlit_app.py           # Main Streamlit application
│   └── ui_components.py           # Reusable UI components
├── results/                       # Generated analysis results
│   ├── events_with_embeddings_*.parquet
│   ├── explanations_*.txt
│   └── pipeline_summary_*.txt
├── main_oop.py                    # OOP-based command line interface
├── app_oop.py                     # OOP-based Streamlit application
├── requirements.txt               # Python dependencies
├── README_OOP.md                  # This documentation
└──  OOP_CONCEPTS.md               # Comprehensive OOP concepts guide
```

## 📚 Documentation

### **OOP Concepts Guide**
- **[OOP_CONCEPTS.md](./OOP_CONCEPTS.md)** - Comprehensive documentation covering:
  - Core OOP concepts (Encapsulation, Inheritance, Polymorphism, Abstraction)
  - SOLID principles with detailed examples
  - Design patterns implementation
  - Architecture patterns
  - Best practices and trade-offs
  - Real-world implementation examples

### **Key Documentation Sections**
- **Architecture Overview**: High-level system design
- **Design Patterns**: Implementation of 6+ design patterns
- **SOLID Principles**: Clean code principles applied
- **Usage Examples**: Practical implementation examples
- **Extension Guide**: How to add new features

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip package manager
- API keys for external services (optional)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd quant_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### API Keys Setup

The application supports various external services. Configure your API keys in `api_keys.txt`:

```bash
# Example api_keys.txt
GROQ_API = "your_groq_api_key_here"
FIN_HUB = "your_finnhub_api_key_here"
OPEN_AI = "your_openai_api_key_here"
HF_TOKEN = "your_huggingface_token_here"
```

### Command Line Interface

```bash
# Run OOP-based analysis
python main_oop.py
```

### Web Interface

```bash
# Run OOP-based Streamlit app
streamlit run app_oop.py
```

## 🔧 Configuration

### Basic Configuration

```python
config = {
    'ticker': 'TSLA',
    'benchmark': 'SPY',
    'start_date': '2024-10-01',
    'end_date': '2024-12-31',
    'z_threshold': 2.5,
    'vol_window': 10,
    'vol_multiplier': 2.0,
    'news_service': 'finnhub',
    'embedding_service': 'sentence_transformer',
    'ai_service': 'groq'
}
```

### Advanced Configuration

```python
from pipeline.pipeline_factory import PipelineBuilder

# Using Builder pattern
pipeline = (PipelineBuilder()
    .with_ticker('AAPL')
    .with_benchmark('SPY')
    .with_date_range('2024-01-01', '2024-12-31')
    .with_anomaly_detection(z_threshold=3.0, vol_window=15)
    .with_news_service('finnhub', api_key='your_key')
    .with_ai_service('groq', api_key='your_key')
    .with_console_observer()
    .build())
```

## 🎯 Key Features

### 1. **Comprehensive OOP Implementation**
- **Encapsulation**: Data hiding and controlled access
- **Inheritance**: Code reuse and polymorphism
- **Abstraction**: Clear interfaces and contracts
- **Polymorphism**: Runtime method resolution

### 2. **SOLID Principles**
- **Single Responsibility**: Each class has one purpose
- **Open/Closed**: Extensible without modification
- **Liskov Substitution**: Interchangeable implementations
- **Interface Segregation**: Focused, specific interfaces
- **Dependency Inversion**: Abstractions over concretions

### 3. **Design Patterns**
- **Strategy Pattern**: Interchangeable algorithms
- **Factory Pattern**: Object creation abstraction
- **Observer Pattern**: Event-driven updates
- **Template Method**: Algorithm structure definition
- **Builder Pattern**: Complex object construction
- **Repository Pattern**: Data access abstraction

### 4. **Strategy Pattern Implementation**
- **Data Loading**: Yahoo Finance, custom APIs
- **Anomaly Detection**: Z-score, Isolation Forest
- **News Retrieval**: Finnhub, Yahoo Finance
- **Embeddings**: Sentence Transformers, OpenAI
- **Similarity Analysis**: FAISS, Cosine Similarity
- **AI Explanations**: Groq, OpenAI

### 5. **Observer Pattern for Progress Tracking**
```python
from pipeline.progress_observer import ConsoleProgressObserver, StreamlitProgressObserver

# Console observer
console_observer = ConsoleProgressObserver()

# Streamlit observer
streamlit_observer = StreamlitProgressObserver(progress_bar, status_text, results_container)
```

### 6. **Factory Pattern for Component Creation**
```python
from processors.news_retriever import NewsRetrieverFactory
from processors.embedding_generator import EmbeddingGeneratorFactory
from processors.ai_explainer import AIExplainerFactory

# Create components
news_retriever = NewsRetrieverFactory.create_retriever('finnhub', api_key)
embedding_generator = EmbeddingGeneratorFactory.create_generator('sentence_transformer')
ai_explainer = AIExplainerFactory.create_explainer('groq', api_key)
```

### 7. **Template Method Pattern for Pipeline Execution**
```python
class QuantitativeAnalysisPipeline(BaseAnalysisPipeline):
    def _preprocess(self, **kwargs):
        # Data loading and preprocessing
        pass
    
    def _analyze(self, **kwargs):
        # Anomaly detection, news retrieval, embeddings, similarity analysis
        pass
    
    def _postprocess(self, **kwargs):
        # Result saving and reporting
        pass
```

### 8. **Repository Pattern for Data Persistence**
```python
from core.base import BaseDataRepository

repository = BaseDataRepository()
repository.save_events(events, 'events.parquet')
repository.save_explanations(explanations, 'explanations.txt')
```

## 🔍 Usage Examples

### Basic Analysis

```python
from pipeline.pipeline_factory import PipelineDirector

# Create basic pipeline
pipeline = PipelineDirector.create_basic_pipeline('TSLA', 'SPY')

# Run analysis
results = pipeline.run()
```

### Advanced Analysis with Custom Configuration

```python
from pipeline.pipeline_factory import PipelineBuilder

# Build custom pipeline
pipeline = (PipelineBuilder()
    .with_ticker('AAPL')
    .with_benchmark('SPY')
    .with_anomaly_detection(z_threshold=3.0)
    .with_news_service('finnhub', api_key='your_key')
    .with_ai_service('groq', api_key='your_key')
    .with_console_observer()
    .build())

# Run analysis
results = pipeline.run()
```

### Custom Strategy Implementation

```python
from processors.anomaly_detector import AnomalyDetectionStrategy

class CustomAnomalyStrategy(AnomalyDetectionStrategy):
    def detect(self, data, **kwargs):
        # Custom anomaly detection logic
        pass
    
    def get_parameters(self):
        return {'custom_param': 'value'}
    
    def set_parameters(self, **kwargs):
        # Set custom parameters
        pass

# Use custom strategy
detector = AnomalyDetector(CustomAnomalyStrategy())
```

## 🎨 UI Components

### Reusable UI Components

```python
from ui.ui_components import UIComponentFactory, UIComponentManager

# Create component manager
manager = UIComponentManager()

# Register components
manager.register_component("data_display", UIComponentFactory.create_data_display())
manager.register_component("events_display", UIComponentFactory.create_events_display())
manager.register_component("chart", UIComponentFactory.create_chart())

# Render components
manager.render_component("data_display", data=df, title="Data Summary")
```

## 🔧 Extending the Framework

### Adding New Strategies

1. **Create Strategy Class**:
```python
class NewDataLoadingStrategy(DataLoadStrategy):
    def load_data(self, ticker, start_date, end_date):
        # Implementation
        pass
```

2. **Update Factory**:
```python
class DataLoaderFactory:
    @staticmethod
    def create_new_loader():
        return DataLoader(NewDataLoadingStrategy())
```

3. **Use in Pipeline**:
```python
pipeline = PipelineBuilder().with_data_loader('new').build()
```

### Adding New UI Components

1. **Create Component Class**:
```python
class CustomUIComponent(UIComponent):
    def render(self, **kwargs):
        # Implementation
        pass
```

2. **Register with Manager**:
```python
manager.register_component("custom", CustomUIComponent())
```

## ✅ Current Status

### **Fully Functional Features**
- ✅ **Command Line Interface**: Complete pipeline execution
- ✅ **Streamlit Web Interface**: Interactive web application
- ✅ **Data Loading**: Yahoo Finance integration
- ✅ **Anomaly Detection**: Z-score and Isolation Forest strategies
- ✅ **News Retrieval**: Finnhub and Yahoo Finance strategies
- ✅ **Embedding Generation**: Sentence Transformers integration
- ✅ **Similarity Analysis**: FAISS-based similarity search
- ✅ **AI Explanations**: Groq API integration with detailed analysis
- ✅ **Progress Tracking**: Real-time progress updates
- ✅ **Result Persistence**: Parquet and text file outputs
- ✅ **Error Handling**: Comprehensive exception management
- ✅ **Logging**: Detailed logging throughout the application

### **Tested and Verified**
- ✅ **End-to-End Pipeline**: Complete analysis workflow
- ✅ **API Integrations**: All external services working
- ✅ **UI Components**: Both CLI and web interfaces functional
- ✅ **Data Persistence**: Results saving and loading
- ✅ **Error Recovery**: Graceful error handling and recovery

## 🧪 Testing

### Unit Testing

```python
import unittest
from processors.data_loader import DataLoader, YFinanceDataStrategy

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.loader = DataLoader(YFinanceDataStrategy())
    
    def test_load_data(self):
        data = self.loader.load_market_data('AAPL', 'SPY', '2024-01-01', '2024-01-31')
        self.assertIsNotNone(data)
```

### Integration Testing

```python
def test_pipeline_integration():
    pipeline = PipelineDirector.create_basic_pipeline('AAPL', 'SPY')
    results = pipeline.run()
    assert 'events' in results
    assert 'event_summary' in results
```

### Manual Testing

```bash
# Test command line interface
python main_oop.py

# Test web interface
streamlit run app_oop.py

# Test with different configurations
python main_oop.py --ticker AAPL --benchmark SPY --z_threshold 3.0
```

## 📊 Performance Considerations

### Memory Management
- Lazy loading of large datasets
- Efficient embedding storage
- Garbage collection optimization

### Scalability
- Parallel processing for embeddings
- Batch processing for large datasets
- Caching for repeated operations

### Error Handling
- Comprehensive exception hierarchy
- Graceful degradation
- Detailed logging

## 🔒 Security

### API Key Management
- Environment variable support
- Secure key storage
- Key rotation support

### Data Privacy
- Local processing by default
- Optional cloud services
- Data encryption

## 📈 Monitoring and Logging

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analysis.log')
    ]
)
```

### Progress Tracking
```python
from pipeline.progress_observer import ConsoleProgressObserver

observer = ConsoleProgressObserver()
pipeline.add_observer(observer)
```

## 🚀 Deployment

### Docker Support
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main_oop.py"]
```

## 🤝 Contributing

### Code Style
- Follow PEP 8
- Use type hints
- Document all public methods
- Write unit tests

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request
