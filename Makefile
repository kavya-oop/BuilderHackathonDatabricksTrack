IMAGE ?= shopping-sessions:latest
PLATFORM ?= linux/amd64
CPUS ?= 8
MEM ?= 7g

.PHONY: build fixtures run-dev check-dev

build:
	docker buildx build --platform $(PLATFORM) --load -t $(IMAGE) .

fixtures:
	.tools/venv/bin/python scripts/run_fixtures.py starter-polars/fixtures -- \
		./scripts/run_fixture_container.sh $(IMAGE) $(CPUS) $(MEM)

run-dev:
	mkdir -p output scratch
	docker run --rm --network none --cpus $(CPUS) --memory $(MEM) \
		--read-only --tmpfs /tmp:rw,size=64m --user 1000:1000 \
		-e DATA_DIR=/data -e OUTPUT_DIR=/output -e SCRATCH_DIR=/scratch \
		-e NCPU=$(CPUS) -e MEM_GB=7 \
		-v "$$(pwd)/dev:/data:ro" -v "$$(pwd)/output:/output" \
		-v "$$(pwd)/scratch:/scratch" $(IMAGE)

check-dev:
	.tools/venv/bin/python scripts/compare.py output dev-answer
