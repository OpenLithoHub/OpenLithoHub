.PHONY: check-plugins

check-plugins:
	python3 -c "from openlithohub.plugins import list_plugins; status = list_plugins(); print('Plugin status:', status); assert isinstance(status, dict); print('OK: plugin infrastructure healthy')"
