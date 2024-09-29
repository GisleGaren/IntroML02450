from invoke import task

@task
def build(ctx):
    ctx.run('latexmk -pdf table.tex')
    ctx.run('latexmk -pdf main.tex')

@task
def clean(ctx):
    ctx.run('latexmk -c main.tex')
