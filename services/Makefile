PROTO_DIR := proto
GEN_DIR := generated
PYTHON := python

.PHONY: all test proto-gen clean install start-extraction stop-extraction start-embedding stop-embedding start-training stop-training start-all stop-all

all: proto-gen start-all

test:
	$(PYTHON) -m pytest tests/ -v

# Generate Python gRPC stubs from proto definitions.
proto-gen:
	@mkdir -p $(GEN_DIR)/persephone/extraction
	@mkdir -p $(GEN_DIR)/persephone/embedding
	@mkdir -p $(GEN_DIR)/persephone/common
	@mkdir -p $(GEN_DIR)/hades/training
	$(PYTHON) -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(GEN_DIR) \
		--grpc_python_out=$(GEN_DIR) \
		$(PROTO_DIR)/persephone/common/common.proto \
		$(PROTO_DIR)/persephone/extraction/extraction.proto \
		$(PROTO_DIR)/persephone/embedding/embedding.proto \
		$(PROTO_DIR)/hades/training/training.proto
	@# Create __init__.py files for package imports
	@touch $(GEN_DIR)/__init__.py
	@touch $(GEN_DIR)/persephone/__init__.py
	@touch $(GEN_DIR)/persephone/common/__init__.py
	@touch $(GEN_DIR)/persephone/extraction/__init__.py
	@touch $(GEN_DIR)/persephone/embedding/__init__.py
	@touch $(GEN_DIR)/hades/__init__.py
	@touch $(GEN_DIR)/hades/training/__init__.py
	@echo "Proto stubs generated in $(GEN_DIR)/"

clean:
	rm -rf $(GEN_DIR)

install:
	pip install -e .

start-extraction:
	systemctl start hades-extractor

stop-extraction:
	systemctl stop hades-extractor

start-embedding:
	systemctl start hades-embedder

stop-embedding:
	systemctl stop hades-embedder

start-training:
	systemctl start hades-trainer

stop-training:
	systemctl stop hades-trainer

start-all: start-extraction start-embedding

stop-all: stop-extraction stop-embedding
