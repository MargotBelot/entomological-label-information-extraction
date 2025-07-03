.PHONY: build clean

build:
	docker compose up --build -d --remove-orphans

clean:
	docker container prune -f