# Use a slim Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /memory_streamlit

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["python3", "-m", "streamlit", "run", "memory_streamlit.py"]