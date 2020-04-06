from typing import TYPE_CHECKING, Iterable, Tuple

if TYPE_CHECKING:
    import graphviz  # noqa: F401


class GraphStyle:
    def __init__(self,
                 node_shape='box',
                 node_fontsize=10,
                 node_draw_ports=False,
                 node_color='#a1c9f4',
                 node_font_color='black',
                 node_outline_color='#4878d0',
                 node_style='filled, solid',

                 bg_color='white',
                 fontname="Verdana",

                 transform_shape='component',
                 transform_color='#ffb482',
                 transform_outline_color='#dd8452',
                 transform_font_color='black',

                 subgraph_cluster=True,
                 subgraph_style='solid',
                 subgraph_color='lightgray',
                 subgraph_outline_color='gray',
                 subgraph_even_color='white',
                 subgraph_even_outline_color='white',
                 subgraph_fontsize=10,
                 subgraph_just='l',

                 edge_color='gray',
                 edge_label_ports=True,
                 edge_fontsize=10,

                 io_fontsize=10,
                 input_shape='cds',
                 input_color='#dfdfdf',
                 input_outline_color='#dfdfdf',
                 input_font_color='black',
                 input_style='filled, solid',

                 output_shape='cds',
                 output_color='#dfdfdf',
                 output_outline_color='#dfdfdf',
                 output_font_color='black',
                 output_style='filled, solid',

                 ):
        self.node_shape = node_shape
        self.node_fontsize = node_fontsize
        self.node_draw_ports = node_draw_ports
        self.node_color = node_color
        self.node_font_color = node_font_color
        self.node_outline_color = node_outline_color
        self.node_style = node_style

        self.fontname = fontname
        self.bg_color = bg_color

        self.transform_shape = transform_shape
        self.transform_color = transform_color
        self.transform_outline_color = transform_outline_color
        self.transform_font_color = transform_font_color

        self.subgraph_cluster = subgraph_cluster
        self.subgraph_style = subgraph_style
        self.subgraph_color = subgraph_color
        self.subgraph_outline_color = subgraph_outline_color
        self.subgraph_even_color = subgraph_even_color
        self.subgraph_even_outline_color = subgraph_even_outline_color
        self.subgraph_fontsize = subgraph_fontsize
        self.subgraph_just = subgraph_just

        self.edge_color = edge_color
        self.edge_label_ports = edge_label_ports
        self.edge_fontsize = edge_fontsize

        self.io_fontsize = io_fontsize
        self.input_shape = input_shape
        self.input_color = input_color
        self.input_outline_color = input_outline_color
        self.input_font_color = input_font_color
        self.input_style = input_style

        self.output_shape = output_shape
        self.output_color = output_color
        self.output_outline_color = output_outline_color
        self.output_font_color = output_font_color
        self.output_style = output_style


DEFAULT_GRAPH_STYLE = 'default'
GRAPH_STYLES = {
    DEFAULT_GRAPH_STYLE: GraphStyle(),
}

GRAPH_STYLE = GRAPH_STYLES[DEFAULT_GRAPH_STYLE]


def set_style(style: str):
    global GRAPH_STYLE
    if isinstance(style, str):
        if style not in GRAPH_STYLES:
            raise ValueError("{} is not valid style. choose one of [{}]".format(style, list(GRAPH_STYLES.keys())))
        GRAPH_STYLE = GRAPH_STYLES[style]


class GraphVizRenderable:

    def graphviz_render(self, g, path):
        raise NotImplementedError()

    def show(self):
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("Graphviz not installed") from None

        g = Digraph('G', engine='dot', graph_attr=[
            ('bgcolor', GRAPH_STYLE.bg_color),
        ])

        self.graphviz_render(g, '')

        return g


class GraphvizNodeMixin(GraphVizRenderable):

    @property
    def graphviz_show_input_ports(self):
        return len(self._graphviz_input_names) > 1

    @property
    def graphviz_show_output_ports(self):
        return len(self._graphviz_output_names) > 1

    def _graphviz_node_id(self, path):
        return '{}{}'.format(path, self._graphviz_node_name)

    @property
    def _graphviz_node_name(self):
        raise NotImplementedError()

    def graphviz_input_node_id(self, input_slot, path):
        return self._graphviz_node_id(path)

    def graphviz_input_port_id(self, input_slot):
        if self.graphviz_show_input_ports:
            return 'inp_{}'.format(input_slot)

    def graphviz_input_port_name(self, input_slot):
        if self.graphviz_show_input_ports:
            return input_slot

    def graphviz_output_node_id(self, output_slot, path):
        return self._graphviz_node_id(path)

    def graphviz_output_port_id(self, output_slot):
        if self.graphviz_show_output_ports:
            return 'out_{}'.format(output_slot)

    def graphviz_output_port_name(self, output_slot):
        if self.graphviz_show_output_ports:
            return output_slot

    @property
    def _graphviz_input_names(self):
        raise NotImplementedError()

    @property
    def _graphviz_output_names(self):
        raise NotImplementedError()

    def graphviz_render(self, g, path):
        """
        :type g: graphviz.Digraph
        """
        global GRAPH_STYLE

        if GRAPH_STYLE.node_draw_ports:
            input_defs = ''.join([
                '<TD PORT="{port_id}">{port_name}</TD>'.format(
                    port_id=self.graphviz_input_port_id(inp),
                    port_name=self.graphviz_input_port_name(inp)
                )
                for inp in self._graphviz_input_names if self.graphviz_input_port_id(inp)
            ])
            if input_defs:
                input_defs = '<TR> {} </TR>'.format(input_defs)

            output_defs = ''.join([
                '<TD PORT="{port_id}">{port_name}</TD>'.format(
                    port_id=self.graphviz_output_port_id(out),
                    port_name=self.graphviz_output_port_name(out)
                )
                for out in self._graphviz_output_names if self.graphviz_output_port_id(out)
            ])

            if output_defs:
                output_defs = '<TR> {} </TR>'.format(output_defs)
            max_span = max(len(self._graphviz_input_names), len(self._graphviz_output_names))
        else:
            max_span = 1
            input_defs = ''
            output_defs = ''

        attrs = [
            ('shape', GRAPH_STYLE.node_shape),
            ('style', GRAPH_STYLE.node_style),
            ('fillcolor', GRAPH_STYLE.node_color),
            ('width', '1.5'),
            ('margin', '0.1'),
            ('fontsize', str(GRAPH_STYLE.node_fontsize)),
            ('fontname', GRAPH_STYLE.fontname),
            ('fontcolor', GRAPH_STYLE.node_font_color),
            ('color', GRAPH_STYLE.node_outline_color),
        ]

        if GRAPH_STYLE.node_draw_ports:
            return g.node(self._graphviz_node_id(path), f'''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" STYLE="ROUNDED">
              {input_defs}
              <TR>
                <TD COLSPAN="{max_span}">{self._graphviz_node_name}</TD>
              </TR>
              {output_defs}
            </TABLE>>''',
                          _attributes=attrs)
        else:
            return g.node(self._graphviz_node_id(path), self._graphviz_node_name,
                          _attributes=attrs
                          )


class GraphvizEdgeMixin(GraphVizRenderable):
    # =======================================================
    # Graphviz representation
    # =======================================================

    @property
    def _graphviz_upstream(self) -> GraphvizNodeMixin:
        raise NotImplementedError()

    @property
    def _graphviz_downstream(self) -> GraphvizNodeMixin:
        raise NotImplementedError()

    @property
    def _graphviz_upstream_slot(self):
        raise NotImplementedError()

    @property
    def _graphviz_downstream_slot(self):
        raise NotImplementedError()

    def graphviz_render(self, g, path):
        """
        :type g: graphviz.Digraph
        """
        output_port = ''
        input_port = ''

        additional_attributes = []
        if GRAPH_STYLE.node_draw_ports:
            _output_port = self._graphviz_upstream.graphviz_output_port_id(self._graphviz_upstream_slot)
            if _output_port:
                output_port = ':' + self._graphviz_upstream.graphviz_output_port_id(self._graphviz_upstream_slot)

            _input_port = self._graphviz_downstream.graphviz_input_port_id(self._graphviz_downstream_slot)
            if _input_port:
                input_port = ':' + self._graphviz_downstream.graphviz_input_port_id(self._graphviz_downstream_slot)
        elif GRAPH_STYLE.edge_label_ports:
            _output_port = self._graphviz_upstream.graphviz_output_port_name(self._graphviz_upstream_slot)
            if _output_port:
                additional_attributes.append(('taillabel', _output_port))
            _input_port = self._graphviz_downstream.graphviz_input_port_name(self._graphviz_downstream_slot)
            if _input_port:
                additional_attributes.append(('headlabel', _output_port))
        return g.edge(
            self._graphviz_upstream.graphviz_output_node_id(self._graphviz_upstream_slot, path) + output_port,
            self._graphviz_downstream.graphviz_input_node_id(self._graphviz_downstream_slot, path) + input_port,
            _attributes=[
                            ('color', GRAPH_STYLE.edge_color),
                            ('fontsize', str(GRAPH_STYLE.node_fontsize)),
                            ('fontname', str(GRAPH_STYLE.fontname)),
                        ] + additional_attributes
        )


class GraphvizGraphMixin(GraphVizRenderable):

    def graphviz_get_input_id(self, input_slot, path):
        return '{}inp_{}'.format(path, input_slot)

    def graphviz_get_output_id(self, output_slot, path):
        return '{}out_{}'.format(path, output_slot)

    def graphviz_add_input(self,
                           g,
                           input_name: str,
                           downstream_node: GraphvizNodeMixin,
                           downstream_slot: str,
                           path: str):

        # Create node
        g.node(self.graphviz_get_input_id(input_name, path), input_name, shape=GRAPH_STYLE.input_shape,
               _attributes=[
                   ('fontsize', str(GRAPH_STYLE.io_fontsize)),
                   ('fontname', str(GRAPH_STYLE.fontname)),
                   ('color', GRAPH_STYLE.input_outline_color),
                   ('fontcolor', GRAPH_STYLE.input_font_color),
                   ('fillcolor', GRAPH_STYLE.input_color),
                   ('style', GRAPH_STYLE.input_style),
               ])

        # Connect node
        downstream_node_name = downstream_node.graphviz_input_node_id(downstream_slot, path)
        downstream_node_port = ''
        additional_attributes = []
        if GRAPH_STYLE.node_draw_ports:
            _downstream_node_port = downstream_node.graphviz_input_port_id(downstream_slot)
            if _downstream_node_port:
                downstream_node_port = ':' + _downstream_node_port
        elif GRAPH_STYLE.edge_label_ports:
            _downstream_node_port = downstream_node.graphviz_input_port_name(downstream_slot)
            if _downstream_node_port:
                additional_attributes.append(('headlabel', _downstream_node_port))
        g.edge(self.graphviz_get_input_id(input_name, path), downstream_node_name + downstream_node_port,
               _attributes=[
                               ('color', GRAPH_STYLE.edge_color),
                               ('fontsize', str(GRAPH_STYLE.edge_fontsize)),
                               ('fontname', str(GRAPH_STYLE.fontname)),
                           ] + additional_attributes)

    def graphviz_add_output(self,
                            g,
                            output_name: str,
                            upstream_node: GraphvizNodeMixin,
                            upstream_slot: str,
                            path: str):

        # Create node
        output_node_id = self.graphviz_get_output_id(output_name, path)

        with g.subgraph(name="cluster_" + path + "_outputs") as sg:
            sg.attr(label="")
            sg.attr(style="invis")

            sg.node(output_node_id, output_name, shape=GRAPH_STYLE.output_shape,
                    _attributes=[
                        ('fontsize', str(GRAPH_STYLE.io_fontsize)),
                        ('fontname', str(GRAPH_STYLE.fontname)),
                        ('orientation', str(180)),
                        ('color', GRAPH_STYLE.output_outline_color),
                        ('fontcolor', GRAPH_STYLE.output_font_color),
                        ('fillcolor', GRAPH_STYLE.output_color),
                        ('style', GRAPH_STYLE.output_style),
                    ])

        # # Connect node
        upstream_node_name = upstream_node.graphviz_output_node_id(upstream_slot, path)
        upstream_node_port = ''
        additional_attributes = []
        if GRAPH_STYLE.node_draw_ports:
            _upstream_node_port = upstream_node.graphviz_output_port_id(upstream_slot)
            if _upstream_node_port:
                upstream_node_port = ':' + _upstream_node_port
        elif GRAPH_STYLE.edge_label_ports:
            _downstream_node_port = upstream_node.graphviz_input_port_name(upstream_slot)
            if _downstream_node_port:
                additional_attributes.append(('taillabel', _downstream_node_port))
        g.edge(upstream_node_name + upstream_node_port, output_node_id,
               _attributes=[
                               ('color', GRAPH_STYLE.edge_color),
                               ('fontsize', str(GRAPH_STYLE.edge_fontsize)),
                               ('fontname', str(GRAPH_STYLE.fontname)),

                           ] + additional_attributes)

    @property
    def _graphviz_vertices(self) -> Iterable[GraphvizNodeMixin]:
        raise NotImplementedError()

    @property
    def _graphviz_edges(self) -> Iterable[GraphvizEdgeMixin]:
        raise NotImplementedError()

    @property
    def _graphviz_inputs(self) -> Iterable[Tuple[str, GraphvizNodeMixin, str]]:
        """
        :return: [ (name, downstream_node, downstream_slot), ... ]
        """
        raise NotImplementedError()

    @property
    def _graphviz_outputs(self) -> Iterable[Tuple[str, GraphvizNodeMixin, str]]:
        """
        :return: [ (name, upstream_node, upstream_slot), ... ]
        """
        raise NotImplementedError()

    def graphviz_render(self, g, path):

        # Render vertices and edges
        for vertex in self._graphviz_vertices:
            vertex.graphviz_render(g, path)

        for edge in self._graphviz_edges:
            edge.graphviz_render(g, path)

        # Render inputs
        for input_name, downstream_node, downstream_slot in self._graphviz_inputs:
            self.graphviz_add_input(g, input_name, downstream_node, downstream_slot, path)

        # Render outputs
        for output_name, upstream_node, upstream_slot in self._graphviz_outputs:
            self.graphviz_add_output(g, output_name, upstream_node, upstream_slot, path)
