.DEFAULT_GOAL := help

TEST_COMMAND = pytest
help:
	@echo 'Use "make test" to run all the unit tests and docstring tests.'
	@echo 'Use "make pep8" to validate PEP8 compliance.'
	@echo 'Use "make html" to create html documentation with sphinx'
	@echo 'Use "make all" to run all the targets listed above.'
test:
	$(TEST_COMMAND) scripts
	$(TEST_COMMAND)
pep8:
	pycodestyle deepblast setup.py
	flake8 songbird setup.py scripts scripts/deepblast-train.py

all: pep8 test
