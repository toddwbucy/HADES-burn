PROTO_DIR := proto
GEN_DIR := generated
# Prefer this project's venv interpreter when present so targets work from a
# cold shell (no activated venv); override with `make PYTHON=python ...`.
PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: all test proto-gen clean install install-units install-embedder install-extractor install-trainer start-extraction stop-extraction start-embedding stop-embedding start-training stop-training start-all stop-all

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
	$(PYTHON) -m pip install -e .

# Install a service systemd unit + its config, substituting this repo's
# services/ path so the unit isn't pinned to any one workstation. Run from
# services/. The Rust-side daemon unit needs no substitution and is installed
# by scripts/install/hades-install.sh.
#
# $(1) = service name (embedder|extractor|trainer), $(2) = config file mode
define install_unit
	sed 's|@SERVICES_DIR@|$(CURDIR)|g' systemd/hades-$(1).service \
		| sudo tee /etc/systemd/system/hades-$(1).service > /dev/null
	sudo chmod 644 /etc/systemd/system/hades-$(1).service
	sudo install -D -m $(2) systemd/$(1).conf /etc/hades/$(1).conf
	sudo systemctl daemon-reload
	@echo "Installed hades-$(1).service (SERVICES_DIR=$(CURDIR))"
endef

install-embedder:
	$(call install_unit,embedder,644)

install-extractor:
	$(call install_unit,extractor,644)

install-trainer:
	$(call install_unit,trainer,644)

install-units: install-embedder install-extractor install-trainer

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
