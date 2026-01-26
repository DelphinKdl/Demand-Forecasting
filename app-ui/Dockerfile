# Use an official lightweight Python image
FROM python:3.11-slim

# Set global environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app


# Set the working directory
WORKDIR /app

# Install compilers and development libraries needed to build and 
# install Python packages (e.g., numpy, pandas, matplotlib) 
#with C/C++ extensions for ML and data science.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        python3-dev \
        libfreetype6-dev \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker's layer cache
COPY app-ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and shared code
COPY app-ui/ .
COPY common/ ./common/
COPY app-ml/src/ ./app-ml/src/

# Expose the port the app runs on
EXPOSE 8050

# Set the default command to run the Dash app using a production server
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8050"]