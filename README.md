URIEL+:Expanding Feature Coverage and Improving Usability of URIEL
======
URIEL+ is expanded upon the original URIEL knowledge base from the paper [(Littell et al., EACL 2017)](https://aclanthology.org/E17-2002/), focusing on describing languages distances through typological vectors, which has been cited over 200+ times. This expansion addresses previous limitaions in feature coverage and usability, particularly focusing on improving support for low-resource languages.

Key Features
-----------
1. **Improved Language Coverage:** URIEL+ integrates five additional databases, including Grambank, BDPROTO, APiCS, and eWAVE, significantly enhancing its typological feature coverage and adds new orphological data for nearly 2500 languages. 
2.   **Advanced Imputation Method:** alongside the original kNN imputation, URIEL+ provided interface for MIDASpy and SoftImpute, providing more accurate imputed data for missing values.
3.   **Customizable Fetaure Selection:** users can choose or exclude specific features when calculating linguistic distances, allowing for tailed analyses depending on your usecase.
4.   **Improved Usability:** Instead do precomputed distances, URIEL+ computes distances dynamically ensuring they reflect the most current data. Each calculated distance is also accompnaied by a confidence score, helping users assess the reliability of the results.

Applications
------------
URIEL+ has evaluated across several downstream tasks, including performance prdiction(PerfPred, ProxyLM), transfer language selection (LangRank), and typological feature-driven language analysis (LinguAlchemy), where it always demonstrates a performance on par to URIEL, if not better.



Installation
------------
The data are store in an `npz` file format, which comes out to be larger than github's size limit. Hence you will have to manually run the `update_ALL()` function in URIEL to First clone the repository and run `setup.py`.
~~~
git clone https://github.com/Masonshipton25/URIELPlusPlus
cd URIELPlusPlus
python3 setup.py install
~~~
Important Note: Upon cloning this repo locally, then immediately under your repository root, make sure to download glottolog v5.0 from https://zenodo.org/records/10804357
