---
layout: post
title: "Xerus"
subtitle: "a general purpose tensor library"
---

# Xerus
**master:** <a href="https://git.hemio.de/xerus/xerus/commits/master"><img alt="build status" src="https://git.hemio.de/xerus/xerus/badges/master/build.svg" /></a>
<a href="https://git.hemio.de/xerus/xerus/commits/master"><img alt="coverage report" src="https://git.hemio.de/xerus/xerus/badges/master/coverage.svg" /></a>
<a href="https://git.hemio.de/xerus/xerus/commits/master">
__breakFix1
{% include version.svp %}
__breakFix2
**development:**  <a href="https://git.hemio.de/xerus/xerus/commits/development"><img alt="build status" src="https://git.hemio.de/xerus/xerus/badges/development/build.svg" /></a>
<a href="https://git.hemio.de/xerus/xerus/commits/development"><img alt="coverage report" src="https://git.hemio.de/xerus/xerus/badges/development/coverage.svg" /></a>


The `xerus` library is a general purpose library for numerical calculations with higher order tensors, Tensor-Train Decompositions / Matrix Product States and general Tensor Networks.
The focus of development was the simple usability and adaptibility to any setting that requires higher order tensors or decompositions thereof.

The key features include:
* Modern code and concepts incorporating many features of the `C++11` standard.
* Full python bindings with very similar syntax for easy transitions from and to c++.
* Calculation with tensors of arbitrary orders using an intuitive Einstein-like notation `A(i,j) = B(i,k,l) * C(k,j,l);`.
* Full implementation of the Tensor-Train decompositions (MPS) with all neccessary capabilities (including Algorithms like ALS, ADF and CG).
* Lazy evaluation of (multiple) tensor contractions featuring heuristics to automatically find efficient contraction orders.
* Direct integration of the `blas` and `lapack`, as high performance linear algebra backends.
* Fast sparse tensor calculation by usage of the `suiteSparse` sparse matrix capabilities.
* Capabilites to handle arbitrary Tensor Networks.


## Getting Started

`xerus` can be obtained via [git](https://git.hemio.de/xerus/xerus) or as an archived download via the same link.
To build and install the library you might want to check our the [Building xerus](/building_xerus) guide.

Where to go from there depends on what you want to do with `xerus` and how you learn. You can quickly get a feeling for the library
by checking out one of the simpler examples like [QTT](/quickstart). For most people the more elaborate [documentation](/documentation)
will likely be the place to start though. Apart from the basic usage it also explains the mindset behind many design choices and
so is adviced reading material regardless of how deep you want to delve into the library. Finally for the most hardcore developers
out there, there is an elaborate [doxygen documentation](/doxygen).

## Issues

Should you have any problems with the library do not hesitate to contact us at [contact[at]libxerus.org](mailto:contact[at]libxerus.org) or describe your problem in the [issuetracker](https://git.hemio.de/xerus/xerus/issues).


## Licence and Attribution

The `xerus` library is published under the AGPL v3.0. With proper attribution you are thus allowed to freely use and modify the source code but if you distribute your software including the `xerus`
library (or you provide a service based on this library) you have to provide the full source code under a compatible licence. For more information see the [AGPL](http://www.gnu.org/licenses/agpl-3.0.html).

If this library proved useful in your scientific research we would be grateful for any contributions to the source code and citations. A corresponding bibtex entry might look as follows
~~~
@Misc{xerus,
	author = {Huber, Benjamin and Wolf, Sebastian},
	title = {Xerus - A General Purpose Tensor Library},
	howpublished = {\url{https://libxerus.org/}},
	year = {2014--2019}
}
~~~
