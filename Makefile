SHELL := /bin/bash
IMAGE_NAME := ghcr.io/azdevops143/nsightprofiling-gpu:latest
REPORTS_DIR := $(shell pwd)/reports

.PHONY: all build profile stats clean help

all: build profile

help:
	@echo "=================================================================="
	@echo "NVIDIA Nsight Systems Profiling Automation"
	@echo "=================================================================="
	@echo "Targets:"
	@echo "  make build    : Build the multi-stage Docker image with NGC Nsight"
	@echo "  make profile  : Run nsys profile inside container on GPU"
	@echo "  make stats    : Print summary tables from generated .nsys-rep"
	@echo "  make clean    : Remove local build artifacts and reports"
	@echo "=================================================================="

build:
	docker build -t $(IMAGE_NAME) .

profile:
	mkdir -p $(REPORTS_DIR)
	docker run --gpus all --privileged --ipc=host --rm \
		-v $(REPORTS_DIR):/reports \
		$(IMAGE_NAME) 256 1e-4 1000

stats:
	@if [ -f "$(REPORTS_DIR)/heat_diffusion_profile.nsys-rep" ]; then \
		docker run --rm -v $(REPORTS_DIR):/reports $(IMAGE_NAME) \
			nsys stats --report cuda_gpu_kern_sum,cuda_api_sum /reports/heat_diffusion_profile.nsys-rep; \
	else \
		echo "Report file not found. Run 'make profile' first."; \
	fi

clean:
	rm -rf $(REPORTS_DIR)/*.nsys-rep $(REPORTS_DIR)/*.sqlite
