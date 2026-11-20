# Sphinx 文档的简易 Makefile
#

# 这些变量可以在命令行指定。
# 前两个变量也可以通过环境变量设置。
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# 默认目标放在首位，直接执行 make 时显示帮助。
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# 将其他目标交给 Sphinx，使用其 make 模式构建。
# $(O) 是 $(SPHINXOPTS) 的简写。
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
