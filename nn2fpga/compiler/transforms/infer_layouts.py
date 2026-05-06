from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from nn2fpga.compiler.core.tensor_layout import TensorLayout, get_custom_tensor_layout, set_custom_tensor_layout
from nn2fpga.compiler.custom_op.op_base import NN2FPGAOp
from nn2fpga.compiler.core.acceleratorpackage import AcceleratorPackage
from logging import getLogger

logger = getLogger(__name__)

class InferLayouts(Transformation):

    def apply(self, model):
        # Phase 1: propagate from pinned nodes backward and forward
        self._remove_layout_annotations(model)
        for node in model.graph.node:
            op = getCustomOp(node)
            if not isinstance(op, NN2FPGAOp):
                continue
            required = op.accepted_input_layout()
            if required is None:
                continue
            for inp in node.input:
                self._propagate_backward(model, inp, TensorLayout(required))
            produced = op.produced_output_layout(TensorLayout(required))
            for out in node.output:
                self._propagate_forward(model, out, TensorLayout(produced))
        

        # Phase 2: fill unannotated edges with identity
        for vi in list(model.graph.input) + list(model.graph.value_info):
            if get_custom_tensor_layout(model, vi.name) is None and model.get_initializer(vi.name) is None:
                logger.info(f"Tensor does not have layout annotation, assigning identity: {vi.name}")
                rank = len(model.get_tensor_shape(vi.name))
                set_custom_tensor_layout(model, vi.name, TensorLayout.identity(rank))

        # Phase 3: insert transposes where adjacent edges mismatch
        # TODO: not yet implemented — raises on mismatch for now
        self._insert_transposes(model)

        # Update accelerator package metadata to reflect input/output layouts
        ap = AcceleratorPackage.from_json(model.get_metadata_prop("accelerator_package"))
        for value in ap.input_map.values():
            tensor_name = value["new_name"]
            layout = get_custom_tensor_layout(model, tensor_name)
            value["layout"] = layout.get_canonical_name() 
        for value in ap.output_map.values():
            tensor_name = value["new_name"]
            layout = get_custom_tensor_layout(model, tensor_name)
            value["layout"] = layout.get_canonical_name()
        model.set_metadata_prop("accelerator_package", ap.to_json())

        return model, False

    def _propagate_backward(self, model, tensor_name, layout):
        """Walk backward through transparent nodes setting the layout."""
        producer = model.find_producer(tensor_name)
        if producer is None:
            # Reached a graph input
            set_custom_tensor_layout(model, tensor_name, layout)
            logger.info(f"Reached graph input '{tensor_name}', assigning layout {layout}")
            return
        # Stop at reshape boundaries or already-annotated edges
        if get_custom_tensor_layout(model, tensor_name) is not None:
            return
        op = getCustomOp(producer)
        if isinstance(op, NN2FPGAOp) and op.accepted_input_layout() is not None:
            # Hit another pinned node — stop, don't overwrite its contract
            return
        set_custom_tensor_layout(model, tensor_name, layout)
        logger.info(f"Propagating layout {layout} backward from '{tensor_name}'")
        for inp in producer.input:
            self._propagate_backward(model, inp, layout)

    def _propagate_forward(self, model, tensor_name, layout):
        """Walk forward through transparent nodes setting the layout."""
        if get_custom_tensor_layout(model, tensor_name) is not None:
            return
        set_custom_tensor_layout(model, tensor_name, layout)
        logger.info(f"Propagating layout {layout} forward from '{tensor_name}'")
        consumers = model.find_consumers(tensor_name)
        for consumer in consumers:
            op = getCustomOp(consumer)
            if isinstance(op, NN2FPGAOp) and op.accepted_input_layout() is not None:
                # Hit a pinned node — stop
                return
            for out in consumer.output:
                self._propagate_forward(model, out, layout)

    def _insert_transposes(self, model):
        """Insert transpose nodes where adjacent edge layouts mismatch."""
        for node in model.graph.node:
            for inp in node.input:
                producer = model.find_producer(inp)
                if producer is None:
                    continue
                src_layout = get_custom_tensor_layout(model, inp)
                op = getCustomOp(node)
                if isinstance(op, NN2FPGAOp):
                    required = op.accepted_input_layout()
                    if required is None:
                        continue
                    tgt_layout = TensorLayout(required)
                else:
                    continue
                if src_layout != tgt_layout:
                    raise NotImplementedError(
                        f"Layout mismatch at '{node.name}' input '{inp}': "
                        f"got {src_layout}, expected {tgt_layout}. "
                        f"Transpose insertion not yet implemented."
                    )
    
    def _remove_layout_annotations(self, model):
        """Remove all layout annotations from the model."""
        for vi in list(model.graph.input) + list(model.graph.value_info):
            set_custom_tensor_layout(model, vi.name, None)