# Sphinx 文档构建器的配置文件。
#
# 内置配置值的完整列表见文档：
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# 项目信息
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "InsightFlow"
author = "InsightFlow contributors"
release = "1.0.0b1"

# 通用配置
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# 建立文档的密钥依赖性
autodoc_mock_imports = []
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# 设置环境变量以显示我们在建文档
os.environ["SPHINX_BUILD"] = "1"

# MyST 解析器配置
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = []

# Autodoc 配置
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# 用于 Google/ NumPy 样式的 Napoleon 配置
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- HTML 输出的选项 -- -- -- -- -- -- --
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_show_copyright = False
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# 发布文档时由部署环境设置站点地址，避免在源码中写入未经确认的仓库链接。

# RTD 主题选项
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
