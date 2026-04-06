PROTO_DIR := proto
GEN_DIR := generated
PYTHON := python

.PHONY: all test proto-gen clean install start-extraction stop-extraction start-embedding stop-embedding start-all stop-all

all: proto-gen start-all

test:
	$(PYTHON) -m pytest tests/ -v

# Generate Python gRPC stubs from proto definitions.
proto-gen:
	@mkdir -p $(GEN_DIR)/persephone/extraction
	@mkdir -p $(GEN_DIR)/persephone/embedding
	@mkdir -p $(GEN_DIR)/persephone/common
	$(PYTHON) -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(GEN_DIR) \
		--grpc_python_out=$(GEN_DIR) \
		$(PROTO_DIR)/persephone/common/common.proto \
		$(PROTO_DIR)/persephone/extraction/extraction.proto \
		$(PROTO_DIR)/persephone/embedding/embedding.proto
	@# Create __init__.py files for package imports
	@touch $(GEN_DIR)/__init__.py
	@touch $(GEN_DIR)/persephone/__init__.py
	@touch $(GEN_DIR)/persephone/common/__init__.py
	@touch $(GEN_DIR)/persephone/extraction/__init__.py
	@touch $(GEN_DIR)/persephone/embedding/__init__.py
	@echo "Proto stubs generated in $(GEN_DIR)/"

clean:
	rm -rf $(GEN_DIR)

install:
	pip install -e .

start-extraction:
	systemctl --user start hades-extractor

stop-extraction:
	systemctl --user stop hades-extractor

start-embedding:
	systemctl --user start hades-embedder

stop-embedding:
	systemctl --user stop hades-embedder

start-all: start-extraction start-embedding

stop-all: stop-extraction stop-embedding
