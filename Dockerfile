FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face sets PORT to 7860 by default
ENV PORT=7860
EXPOSE 7860

# We don't want PyTorch to complain about OpenMP issues in Docker
ENV KMP_DUPLICATE_LIB_OK="TRUE"

# Start the Flask application
CMD ["python", "app.py"]
