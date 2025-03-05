.PHONY: install
install:
	python3 -m pip install .

.PHONY: dev_install
dev_install:
	python3 -m pip install '.[dev,test]'

.PHONY: lint
lint:
	python3 -m pylint llm_dep_parser/

.PHONY: format
format:
	python3 -m black llm_dep_parser/

.PHONY: test
test:
	python3 -m pytest test/