FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole project
COPY . .

# Expose ports for API and Streamlit
EXPOSE 8000
EXPOSE 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "run.py", "--all"]