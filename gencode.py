
import jupytext
import importlib.util
from jinja2 import Environment, FileSystemLoader

def write_model_code(graph, inputs):
    """create code from jinja template
    graph: Graph model
    inputs: dict with params to template. eg.:
    {'model_name':"mymodel", 'initialization':'xavier', 'layers': list node op, 'params': list node params}
    """
    template_dir = "./"
    env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True, )
    template = env.get_template("code-template.py.jinja")

    inputs['layers'] = []
    inputs['params'] = []
    for node in graph.nodes:
        inputs['layers'].append(node.op)
        if isinstance(node.params, dict):
            inputs['params'].append(dict_to_params(node.params))
        elif isinstance(node.params, str):
            inputs['params'].append(node.params)
        else:
            inputs['params'].append('( )')

    G = graph.build()
    f_input = []
    f_output = []
    for nd in G.nodes:
        if G.out_degree(nd) == 0:
            f_output.append(nd)
        if G.in_degree(nd) == 0:
            f_input.append(nd)

    input_nodes = len(graph.nodes) * [None]
    output_nodes = len(graph.nodes) * [None]
    for edge in graph.edges:
        nd_in, nd_out = edge
        output_nodes[int(nd_in)] = 'int_'+str(nd_in)
        input_nodes[int(nd_out)] = 'int_'+str(nd_in)
    for i, nd in enumerate(graph.nodes):
        if str(nd.id) in f_input:
            input_nodes[i] = 'x'
        if str(nd.id) in f_output:
            output_nodes[i] = 'out'
    inputs['inputs'] = input_nodes
    inputs['outputs'] = output_nodes

    code = template.render(**inputs)
    notebook = to_notebook(code)
    return code, notebook

def to_notebook(code):
    """Converts Python code to Jupyter notebook format."""
    notebook = jupytext.reads(code, fmt="py")
    return jupytext.writes(notebook, fmt="ipynb")

def import_from_file(module_name, filepath):
    """
    Imports a module from file
    module_name: Assigned to the module's __name__ parameter
    filepath: Path to the .py file
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def dict_to_params(di):
    """to print a string from dict to pass as nn.function parameters when generating code
    """
    s = '('
    for k in di.keys():
        s += k + ' = ' + str(di[k]) + ', '
    s = s[:-2] # remove last ', '
    s += ')'
    return s