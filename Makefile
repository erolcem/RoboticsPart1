.PHONY: install test demo benchmark serve docker-build docker-up clean

install:
	pip install -e .[dev]

test:
	pytest tests/ -q

demo:
	sitestate demo --out demo_output

benchmark:
	sitestate benchmark --seeds 5 --out demo_output

serve:
	sitestate serve --project demo_output/project_data

docker-build:
	docker build -t sitestate .

docker-up:
	docker compose up --build

clean:
	rm -rf demo_output .pytest_cache src/sitestate.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
