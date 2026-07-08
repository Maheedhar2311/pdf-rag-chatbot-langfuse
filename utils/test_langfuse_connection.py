from tracing.langfuse_callback import langfuse

print("Langfuse client initialized successfully!")

# Flush any pending events
langfuse.flush()

print("Connection successful.")