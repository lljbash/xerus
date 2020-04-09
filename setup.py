import setuptools

with open("README.md", "r") as f:
	long_description = f.read()

setuptools.setup(
	name="Xerus",
	version="4.0",
	author_email="contact@libxerus.org",
	description="A general purpose tensor library",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://libxerus.org",
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: OS Independent",
	],
	install_requires = [
		"numpy",
	],
)
