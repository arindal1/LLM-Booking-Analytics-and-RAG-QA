# LLM-Powered Booking Analytics & QA System

This project provides a comprehensive solution for processing hotel booking data, extracting business insights, and enabling retrieval-augmented question answering (RAG) through an LLM-powered API. The system combines data analytics, vector-based retrieval with FAISS, and a lightweight language model to answer questions about hotel bookings.

---

## Table of Contents

- [LLM-Powered Booking Analytics \& QA System](#llm-powered-booking-analytics--qa-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation \& Setup](#installation--setup)
  - [Data Collection \& Preprocessing](#data-collection--preprocessing)
  - [Analytics \& Reporting](#analytics--reporting)
  - [Retrieval-Augmented Question Answering (RAG)](#retrieval-augmented-question-answering-rag)
  - [API Endpoints](#api-endpoints)
    - [POST `/analytics`](#post-analytics)
    - [POST `/ask`](#post-ask)
    - [GET `/health`](#get-health)
    - [**Sample Test Queries for `/analytics` Endpoint**](#sample-test-queries-for-analytics-endpoint)
    - [**Sample Test Queries for `/ask` Endpoint**](#sample-test-queries-for-ask-endpoint)
  - [Performance Evaluation](#performance-evaluation)
  - [Deployment](#deployment)
  - [Future Enhancements](#future-enhancements)
  - [Troubleshooting \& Tips](#troubleshooting--tips)
  - [Contact](#contact)
  - [References \& Useful Links](#references--useful-links)
- [Happy Coding 🚀](#happy-coding-)

---

## Overview

The project is designed to:

- **Process hotel booking records**: Clean and preprocess raw data.
- **Extract analytics**: Generate insights such as revenue trends, cancellation rates, geographical distribution of bookings, and booking lead time.
- **Answer questions**: Use vector-based retrieval and a pre-trained language model (DistilGPT2) to answer queries regarding booking data.
- **Expose functionality via REST API**: FastAPI endpoints provide access to analytics, Q&A, and system health checks.

---

## Project Structure

```plaintext
.
├── data
│   └── hotel_bookings.csv
├── images
├── notes
│   └── bookinganalytics.pdf
├── app.py
├── embeddings.npy
├── faiss_index.bin
├── report.docx
└── hotel_bookings_preprocessed.csv
```

- **`data/`**: Contains the raw hotel bookings dataset.
- **`images/`**: Folder to store any generated or reference images.
- **`notes/`**: Documentation or project notes (e.g., PDF reports).
- **`app.py`**: The main FastAPI application.
- **`embeddings.npy` & `faiss_index.bin`**: Artifacts for the FAISS vector store.
- **`report.docx`**: Short report explaining implementation choices & challenges.
- **`hotel_bookings_preprocessed.csv`**: Preprocessed dataset ready for analytics.

---

## Installation & Setup

Follow these steps to get the project running on your machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/llm-booking-analytics.git
   cd llm-booking-analytics
   ```

2. **Set up the virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**

   - Ensure that the `data/hotel_bookings.csv` file is available.
   - Run your preprocessing script (e.g., via a Jupyter Notebook) to generate `hotel_bookings_preprocessed.csv`.

4. **Start the FastAPI server:**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

For more detailed instructions, refer to the [FastAPI documentation](https://fastapi.tiangolo.com/).

---

## Data Collection & Preprocessing

- **Dataset**: A sample hotel bookings dataset (CSV) is used. You may also use other relevant datasets as long as they contain required fields.
- **Preprocessing**:
  - Handle missing values and format inconsistencies.
  - Convert date fields (e.g., `arrival_date`) into datetime objects.
  - Use appropriate data types for numerical fields (e.g., `int8`, `float32`).
- **Storage**: Preprocessed data is saved as `hotel_bookings_preprocessed.csv` and loaded in the application for analytics and QA.

<details>
  <summary>More on Data Cleaning</summary>

  - **Missing Values**: Checked and handled during preprocessing.
  - **Data Types**: Fields such as `is_canceled`, `lead_time`, `adr`, etc., are cast to optimal types for performance.
  - **Aggregation**: Data is grouped (e.g., by month) to compute analytics like revenue trends.
</details>

---

## Analytics & Reporting

The system computes various analytics from the preprocessed data:

- **Revenue Trends**: Aggregated monthly revenue calculated using the arrival date.
- **Cancellation Rate**: Percentage of total bookings that were canceled.
- **Geographical Distribution**: Top 5 countries based on the number of bookings.
- **Booking Lead Time Distribution**: Insights into the lead times for bookings.
- **Additional Analytics**: Easily extendable with further metrics if needed.

Analytics are computed using libraries such as [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/), in [bookinganalytics.pdf](notes/bookinganalytics.pdf).

---

## Retrieval-Augmented Question Answering (RAG)

The RAG system integrates several components:

- **Embeddings**:
  - Uses [SentenceTransformer](https://www.sbert.net/) (`paraphrase-MiniLM-L6-v2`) to compute vector embeddings for each booking record.
  - These embeddings are stored in a FAISS index (`faiss_index.bin` and `embeddings.npy`).

- **Question Answering**:
  - For a given user question, the system computes its embedding and retrieves the top *k* similar booking records.
  - The retrieved records form the context for the LLM prompt.
  - Uses [DistilGPT2](https://huggingface.co/distilgpt2) to generate an answer based on the context.

<details>
  <summary>How It Works</summary>

  1. **Embedding Calculation**: Each booking record is converted to a text string and embedded.
  2. **Vector Store**: FAISS is used to quickly search and retrieve similar records.
  3. **Prompt Formation**: Retrieved records are concatenated to form a context prompt.
  4. **LLM Inference**: The language model generates a response based on the provided prompt.
</details>

---

## API Endpoints

The project exposes three main REST API endpoints:

### POST `/analytics`

- **Description**: Returns the computed analytics including cancellation rate, revenue trends, and top booking countries.
- **Example Response**:

  ```json
  {
      "cancellation_rate": 12.34,
      "monthly_revenue_trend": {
          "2017-07-31": 123456.78,
          "2017-08-31": 234567.89
      },
      "top_countries": {
          "USA": 1500,
          "GBR": 900,
          "FRA": 800
      }
  }
  ```

### POST `/ask`

- **Description**: Accepts a natural language question about the booking data and returns an answer generated by the LLM.
- **Request Body**:

  ```json
  {
      "question": "What is the average revenue per booking in July 2017?"
  }
  ```

- **Example Response**:

  ```json
  {
      "question": "What is the average revenue per booking in July 2017?",
      "answer": "Based on the retrieved records, the average revenue per booking in July 2017 was approximately $XXX.XX."
  }
  ```

### GET `/health`

- **Description**: Health check endpoint that returns the status of the system along with details like the FAISS index size and model status.
- **Example Response**:

  ```json
  {
      "status": "ok",
      "faiss_index_size": 10000,
      "model_loaded": true
  }
  ```

<details>
    <summary>Example Queries</summary>

<br>

Here are some **sample test queries** for your API, along with their expected answers based on the data and functionality of the system:

---

### **Sample Test Queries for `/analytics` Endpoint**

1. **Query:**
   - **Request:** `POST /analytics`
   - **Expected Answer:**
     ```json
     {
       "cancellation_rate": 12.34,
       "monthly_revenue_trend": {
         "2017-07-31": 50000.00,
         "2017-08-31": 52000.00,
         "2017-09-30": 48000.00
       },
       "top_countries": {
         "USA": 1500,
         "GB": 1200,
         "DE": 1000,
         "FR": 800,
         "IT": 600
       }
     }
     ```
   - **Explanation:**
     - The cancellation rate is 12.34% of total bookings.
     - Monthly revenue trends show totals for specific months.
     - Top countries are listed with the count of bookings.

---

### **Sample Test Queries for `/ask` Endpoint**

1. **Query:**
   - **Request:** `POST /ask`  
     ```json
     {
       "question": "Show me total revenue for July 2017."
     }
     ```
   - **Expected Answer:**
     ```json
     {
       "question": "Show me total revenue for July 2017.",
       "answer": "The total revenue for July 2017 is $50,000.00."
     }
     ```
   - **Explanation:**
     - The model uses the question to search for relevant booking data (July 2017 revenue).
   
---

2. **Query:**
   - **Request:** `POST /ask`
     ```json
     {
       "question": "Which locations had the highest booking cancellations?"
     }
     ```
   - **Expected Answer:**
     ```json
     {
       "question": "Which locations had the highest booking cancellations?",
       "answer": "The locations with the highest booking cancellations are: USA (500 cancellations), GB (400 cancellations), and DE (350 cancellations)."
     }
     ```
   - **Explanation:**
     - The model retrieves cancellation data per location and provides the top 3 locations with the most cancellations.

---

3. **Query:**
   - **Request:** `POST /ask`
     ```json
     {
       "question": "What is the average price of a hotel booking?"
     }
     ```
   - **Expected Answer:**
     ```json
     {
       "question": "What is the average price of a hotel booking?",
       "answer": "The average price of a hotel booking is $150.00 per night."
     }
     ```
   - **Explanation:**
     - The model calculates the average daily rate (ADR) of hotel bookings.

---

4. **Query:**
   - **Request:** `POST /ask`
     ```json
     {
       "question": "How many bookings were made in 2019?"
     }
     ```
   - **Expected Answer:**
     ```json
     {
       "question": "How many bookings were made in 2019?",
       "answer": "A total of 12,000 bookings were made in 2019."
     }
     ```
   - **Explanation:**
     - The model searches the data for the number of bookings within the year 2019.

---

5. **Query:**
   - **Request:** `POST /ask`
     ```json
     {
       "question": "What was the booking lead time distribution?"
     }
     ```
   - **Expected Answer:**
     ```json
     {
       "question": "What was the booking lead time distribution?",
       "answer": "The lead time distribution is as follows: 0-30 days (45%), 31-60 days (30%), 61+ days (25%)."
     }
     ```
   - **Explanation:**
     - The model provides a breakdown of lead time distribution from the dataset.


</details>

---

## Performance Evaluation

- **Accuracy**: The accuracy of Q&A responses can be evaluated by comparing generated answers with expected outcomes for a set of test queries.
- **Response Time**: Middleware in `app.py` logs the processing time for each API request.
- **Optimization**: Ensure that the FAISS index is pre-built and stored to speed up retrieval. Use batching and efficient preprocessing where applicable.

<details>
  <summary>Evaluation Metrics</summary>

  - **API Response Time**: Measured via the `X-Process-Time` header.
  - **Retrieval Speed**: Number of vectors in the FAISS index indicates scalability.
  - **LLM Response Quality**: Validate through sample queries and user feedback.
</details>

---

## Deployment

To deploy the system:

1. **Local Deployment**: Run the FastAPI app locally using Uvicorn.
2. **Production Deployment**: Consider deploying with a production-ready ASGI server (e.g., Gunicorn with Uvicorn workers) behind a reverse proxy.
3. **Containerization**: Package the solution with Docker for ease of deployment.


---

## Future Enhancements

- **Real-time Data Updates**: Integrate with a database (SQLite, PostgreSQL) to update analytics as new data arrives.
- **Query History Tracking**: Implement a logging mechanism to track user queries.
- **Additional Endpoints**: For example, a dedicated endpoint for retrieving historical query logs.
- **Enhanced Analytics**: Add more metrics and visualizations based on user needs.

<details>
  <summary>Bonus Features</summary>

  - **Health Check Enhancements**: Extend the `/health` endpoint to verify connectivity with external services.
  - **User Authentication**: Secure API endpoints with authentication and authorization.
</details>

---

## Troubleshooting & Tips

- **Preprocessed Data Not Found**:  
  Ensure that `hotel_bookings_preprocessed.csv` is generated and placed in the project root before starting the server.

- **Dependency Issues**:  
  Verify that all dependencies listed in `requirements.txt` are installed in your virtual environment.

- **Performance Bottlenecks**:  
  For large datasets, consider increasing the chunk size when reading CSVs and optimizing the FAISS index building process.

<details>
  <summary>Common Issues</summary>

  - **FAISS Index Not Loading**: Check that `faiss_index.bin` and `embeddings.npy` exist. If not, the application will automatically compute and save these.
  - **LLM Response Latency**: Fine-tune LLM generation parameters (e.g., `max_length`, `temperature`) if responses are slow.
</details>

---

## Contact

For questions or feedback, please reach out via:
- GitHub: [@arindal1](https://github.com/arindal1)
- LinkedIn: [Arindal Char](https://linkedin.com/in/arindalchar)

---

## References & Useful Links

- **FastAPI Documentation**: [FastAPI](https://fastapi.tiangolo.com/)
- **Hugging Face Transformers**: [Transformers](https://huggingface.co/transformers/)
- **Datasets**: [Sample Hotel Bookings Dataset](https://solvei8-aiml-assignment.s3.ap-southeast-1.amazonaws.com/hotel_bookings.csv)

---

# Happy Coding 🚀