from typing import TYPE_CHECKING

import mldag.style
from mldag.core import mldagbase
from mldag.utils import INSPECT_EMPTY_PARAMETER

if TYPE_CHECKING:
    from graphviz import Digraph  # noqa: F401


class MLDagRenderer:

    def display(self, obj, **parameters):
        raise NotImplementedError()


class GraphvizRenderer(MLDagRenderer):
    def __init__(self, style=None):
        self._style = style

    @property
    def style(self):
        return self._style or mldag.style.current()

    def get_node_id(self, node, path, parameters, mldag_id=True):
        if isinstance(node, mldagbase.MLDagNode) and (self._get_node_depth(path) < parameters['max_mldag_depth']):
            if mldag_id:
                return 'cluster_{}{}'.format(path, node.name)
            else:
                return self.get_inner_node(node.mldag, '{}{}/'.format(path, node.name), parameters)
        return '{}{}'.format(path, node.name)

    def get_node_label(self, node):
        return node.name

    def get_node_port_id(self, node, port_name, input_flag):
        if input_flag:
            return 'inp_{}'.format(port_name)
        else:
            return 'out_{}'.format(port_name)

    def get_node_port_label(self, node, port_name, input_flag):
        return port_name

    def get_node_port_names(self, node, input_flag):
        if input_flag:
            return [i.name for i in node.inputs]
        else:
            return [i.name for i in node.outputs]

    def _render_node_ports_htlm(self, node, input_flag):
        defs = [
            '<TD PORT="{port_id}">{port_name}</TD>'.format(
                port_id=self.get_node_port_id(node, inp, input_flag),
                port_name=self.get_node_port_label(node, inp, input_flag),
            )
            for inp in self.get_node_port_names(node, input_flag) if
            self.get_node_port_id(node, inp, input_flag)
        ]
        if defs:
            return '<TR> {} </TR>'.format(''.join(defs)), len(defs)
        return '', 0

    def _shorten_word(self, x, max_len):
        if not max_len:
            return x
        x = x.split(' ')
        n_words = 2
        while len(' '.join(x[:n_words]) + ('\u2026' if n_words <= len(x) else '')) <= max_len and n_words <= len(x):
            n_words += 1

        initial_len = len(x)
        x = ' '.join(x[:(n_words - 1)])
        if len(x) > max_len:
            if max_len > 2:
                x = x[:(max_len - 2)] + '\u2026'
            else:
                x = x[:max_len]
        elif n_words <= initial_len:
            x = x + "\u2026"
        return x

    def render_node(self, node, g, path, parameters):
        if parameters['show_ports']:
            input_defs, input_defs_len = self._render_node_ports_htlm(node, input_flag=True)
            output_defs, output_defs_len = self._render_node_ports_htlm(node, input_flag=False)
            max_span = max(input_defs_len, output_defs_len)
        else:
            max_span = 1
            input_defs = ''
            output_defs = ''

        attrs = self.style['node']
        mro = [i.__name__ for i in node.__class__.__mro__]
        for class_name in mro:
            if class_name in self.style['node-mro']:
                attrs = self.style['node-mro'][class_name]
                break

        if parameters['show_class']:
            annotation = node.__class__.__name__
            annotation = self._shorten_word(annotation, parameters['class_max_len'])
            class_def = f'''
            <TR>
                <TD COLSPAN="{max_span}">&#xab;{annotation}&#xbb;</TD>
            </TR>
            '''
        else:
            class_def = ''

        return g.node(self.get_node_id(node, path, parameters),
                      f'''<
                        <TABLE BORDER="0" CELLBORDER="{self.style['node']['cellborder']}"
                            CELLSPACING="{self.style['node']['cellspacing']}">
                          {input_defs}
                          {class_def}
                          <TR>
                            <TD COLSPAN="{max_span}">{self.get_node_label(node)}</TD>
                          </TR>
                          {output_defs}
                        </TABLE>>''',
                      **attrs)

    def _get_node_depth(self, path):
        return sum((i == '/' for i in path)) - 1

    def _get_mldag_port_node_id(self, p: mldagbase.MLDagBase, path, port_name, input_flag, parameters):
        if self._get_node_depth(path) == 0 or parameters['show_mldag_io']:
            return '{}{}'.format(path, '{}_{}'.format('inp' if input_flag else 'out', port_name))

    def _render_mldag_port_node(self, g, port_annotation, node_id, port_name, input_flag, parameters):
        style = self.style['mldag-input'] if input_flag else self.style['mldag-output']
        if parameters['show_class'] and port_annotation != INSPECT_EMPTY_PARAMETER:
            annotation = port_annotation.__name__ \
                if isinstance(port_annotation, type) else str(port_annotation)
            annotation = self._shorten_word(annotation, parameters['class_max_len'])
            class_tag = "&#xab;" + annotation + "&#xbb;" + "\n"
        else:
            class_tag = ''
        g.node(node_id, class_tag + port_name, **style)

    def _render_mldag_ports(self, p: mldagbase.MLDagBase, g, path, parameters):
        seen_node_ids = {}
        all_ports = [(i, True) for i in p.inputs] + [(i, False) for i in p.outputs]

        input_cluster = g.subgraph(name="cluster_" + path + "_input", graph_attr={'style': 'invis'})
        output_cluster = g.subgraph(name="cluster_" + path + "_output", graph_attr={'style': 'invis'})
        for port, input_flag in all_ports:
            port_name = port.name
            node_id = self._get_mldag_port_node_id(p, path, port_name, input_flag=input_flag, parameters=parameters)
            if not node_id:
                continue
            if node_id not in seen_node_ids:
                if parameters['cluster_mldag_ports']:
                    with input_cluster if input_flag else output_cluster as cluster:
                        self._render_mldag_port_node(cluster, port.type, node_id,
                                                     port_name, input_flag, parameters)
                    self._render_mldag_port_node(g, port.type, node_id,
                                                 port_name, input_flag, parameters)
            connected_nodes = self.get_node_id_and_port(
                port.downstream_node if input_flag else port.upstream_node,
                path,
                port.downstream_slot if input_flag else port.upstream_slot,
                input_flag, parameters)
            for node2, label2 in connected_nodes:
                self._render_edge(g, node_id, '', node2, label2, not input_flag)

    def _get_mldag_node_id_and_port(self, p: mldagbase.MLDagBase, path, port_name, input_flag, parameters):

        node_port = self._get_mldag_port_node_id(p, path, port_name, input_flag, parameters)
        if node_port:
            return [(node_port, ''), ]

        slots = [i for i in (p.inputs if input_flag else p.outputs) if i.name == port_name]
        rv = []
        for slot in slots:
            slot_rv = self.get_node_id_and_port(slot.downstream_node if input_flag else slot.upstream_node,
                                                path,
                                                slot.downstream_slot if input_flag else slot.upstream_slot,
                                                input_flag, parameters)
            rv.extend(slot_rv)
        return rv

    def get_node_id_and_port(self, node, path, port, input_flag, parameters):
        """
        Function for getting node id and port for edge connection
        Recursive
        """

        depth = self._get_node_depth(path)
        max_mldag_depth = parameters['max_mldag_depth']

        if isinstance(node, mldagbase.MLDagNode) and depth < max_mldag_depth:
            return self._get_mldag_node_id_and_port(node.mldag,
                                                    '{}{}/'.format(path, node.name),
                                                    port, input_flag, parameters)

        node_id = self.get_node_id(node, path, parameters)
        port_name = ''
        label = ''
        if parameters['show_ports']:
            _port_name = self.get_node_port_id(node, port, input_flag)
            if _port_name:
                port_name = ':' + _port_name
        if parameters['show_port_labels']:
            label = port
            # Do not draw input port if node has only one required input port
            if parameters['port_labels_minimal']:
                if input_flag and sum((i.default == INSPECT_EMPTY_PARAMETER for i in node.inputs)) == 1:
                    if any((port == i.name and i.default == INSPECT_EMPTY_PARAMETER for i in node.inputs)):
                        label = ''
                elif not input_flag and len(node.outputs) == 1:
                    label = ''
        return [('{}{}'.format(node_id, port_name), label), ]

    def _render_edge(self, g, upstream_node, upstream_label, downstream_node, downstream_label, flip=False):
        if flip:
            upstream_node, downstream_node = downstream_node, upstream_node
            upstream_label, downstream_label = downstream_label, upstream_label
        return g.edge(
            upstream_node,
            downstream_node,
            taillabel=upstream_label,
            headlabel=downstream_label,
            **self.style['edge'],
        )

    def render_edge(self, edge, g, path, parameters):
        """
        :type g: graphviz.Digraph
        """
        for upstream_node, upstream_label in \
                self.get_node_id_and_port(edge.upstream, path, edge.upstream_slot,
                                          input_flag=False, parameters=parameters):
            for downstream_node, downstream_label in \
                    self.get_node_id_and_port(edge.downstream, path, edge.downstream_slot,
                                              input_flag=True, parameters=parameters):
                self._render_edge(g, upstream_node, upstream_label, downstream_node, downstream_label)

    def render_vertex(self, vertex: mldagbase.NodeBase, g, path, parameters):
        depth = self._get_node_depth(path)
        max_mldag_depth = parameters['max_mldag_depth']
        if isinstance(vertex, mldagbase.MLDagNode) and depth < max_mldag_depth:
            new_path = '{}{}/'.format(path, self.get_node_label(vertex))
            subgraph_style = self.style['subgraph'][self._get_node_depth(path) % len(self.style['subgraph'])]
            with g.subgraph(name=self.get_node_id(vertex, path, parameters), graph_attr=subgraph_style) as sg:
                sg.attr(label='&#xab;' + vertex.__class__.__name__ + '&#xbb; \n' + vertex.name)
                self.render_mldag(vertex.mldag, sg, new_path, parameters)
        elif isinstance(vertex, mldagbase.NodeBase):
            self.render_node(vertex, g, path,
                             parameters=parameters)

    def get_inner_node(self, p: mldagbase.MLDagBase, path, parameters):
        """
        Get any node inside mldag (preferrably close to center)
        """
        node_ranks = dict()

        for node in p.vertices:
            node_ranks[node] = 0

        assert len(node_ranks) > 0

        # Calculate node ranks
        updated_rank = True
        while updated_rank:
            updated_rank = False
            for node, node_rank in node_ranks.items():
                downstream_edges = node.get_downstream()
                for edge in downstream_edges:
                    if node_ranks[edge.downstream] != node_rank + 1:
                        node_ranks[edge.downstream] = node_rank + 1
                        updated_rank = True

        max_rank = max(node_ranks.values())
        min_rank = min(node_ranks.values())
        mid_rank = (max_rank + min_rank) / 2

        closest_node = None
        closest_rank = None
        for node, node_rank in node_ranks.items():
            if closest_rank is None or abs(node_rank - mid_rank) < abs(closest_rank - mid_rank):
                closest_node = node
                closest_rank = node_rank

        return self.get_node_id(closest_node, path, parameters, mldag_id=False)

    def render_vertex_dependencies(self, vertex: mldagbase.NodeBase, g, path, parameters):
        self_mldag = isinstance(vertex, mldagbase.MLDagNode)
        cur_node_id = self.get_node_id(vertex, path, parameters, mldag_id=self_mldag)
        for dep_name, dep in vertex.iter_valid_dependencies():
            dep: mldagbase.NodeBase
            dep_node_id = self.get_node_id(dep, path, parameters, mldag_id=False)
            mldag_id = self.get_node_id(dep, path, parameters, mldag_id=True)
            additional_args = dict()
            if dep_node_id != mldag_id:
                additional_args['ltail'] = mldag_id
            g.edge(dep_node_id,
                   cur_node_id,
                   constraint="false",
                   **self.style['dependency'],
                   **additional_args,
                   )

    def render_mldag(self, p: mldagbase.MLDagBase, g, path, parameters):
        """
        :type g: Digraph
        """
        # Render vertices and edges
        for vertex in p.vertices:
            self.render_vertex(vertex, g, path, parameters)
            self.render_vertex_dependencies(vertex, g, path, parameters)

        for edge in p.edges:
            self.render_edge(edge, g, path, parameters)

        self._render_mldag_ports(p, g, path, parameters)

    def get_initial_graph(self):
        try:
            from graphviz import Digraph  # noqa: F811
        except ImportError:
            raise ImportError("Could not import graphviz. Graphviz not installed!") from None
        return Digraph('G', engine='dot',
                       graph_attr={
                           **{'compound': "true"},
                           **self.style['graph']},
                       node_attr=self.style['graph'],
                       edge_attr=self.style['graph'])

    def display(self, obj,
                show_ports=False,
                show_port_labels=True,
                port_labels_minimal=True,
                show_mldag_io=False,
                show_class=True,
                class_max_len=0,
                cluster_mldag_ports=True,
                max_mldag_depth=-1,
                ):

        parameters = {
            'show_ports': show_ports,
            'show_port_labels': show_port_labels,
            'port_labels_minimal': port_labels_minimal,
            'show_mldag_io': show_mldag_io,
            'show_class': show_class,
            'class_max_len': class_max_len,
            'cluster_mldag_ports': cluster_mldag_ports,
            'max_mldag_depth': max_mldag_depth,
        }

        if isinstance(obj, mldagbase.MLDagBase):
            g = self.get_initial_graph()
            self.render_mldag(obj, g, '/', parameters)
            return g
        else:
            raise NotImplementedError("Rendering objects of class {} not implemented".format(obj))


PIPELINE_RENDERER = GraphvizRenderer()


def display(obj,
            renderer=None,
            show_ports=False,
            show_port_labels=True,
            port_labels_minimal=True,
            show_mldag_io=True,
            show_class=True,
            class_max_len=0,
            cluster_mldag_ports=True,
            max_mldag_depth=-1,
            ):
    return (renderer or PIPELINE_RENDERER).display(obj,
                                                   max_mldag_depth=max_mldag_depth,
                                                   port_labels_minimal=port_labels_minimal,
                                                   show_mldag_io=show_mldag_io,
                                                   show_port_labels=show_port_labels,
                                                   show_ports=show_ports,
                                                   show_class=show_class,
                                                   class_max_len=class_max_len,
                                                   cluster_mldag_ports=cluster_mldag_ports,
                                                   )
