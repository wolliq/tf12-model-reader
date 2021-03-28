import org.tensorflow.*;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TFV1V2Reader {
    public static void main( String[] args ) throws IOException
    {
        // good idea to print the version number, 1.2.0 as of this writing
        System.out.println(TensorFlow.version());
        final int NUM_PREDICTIONS = 1;

        // load the model Bundle
        // String tf1ModelPath = "/home/wolliqeonii/workspace/dev/jsl/tf1-tests/saved_models/1616707109";
        String tfModelPath = "/home/wolliqeonii/Documents/jsl/tf2/bert2";

        try (SavedModelBundle bundle = SavedModelBundle.load(tfModelPath, "serve")) {
            if(bundle.metaGraphDef().hasGraphDef() && bundle.metaGraphDef().getSignatureDefCount() > 0) {
                for (SignatureDef sigDef : bundle.metaGraphDef().getSignatureDefMap().values()) {
                    Map<String, TensorInfo> inputs = sigDef.getInputsMap();
                    for (Map.Entry<String, TensorInfo> e : inputs.entrySet()){
                        String key = e.getKey();
                        TensorInfo tfInfo = e.getValue();
                        System.out.println(
                                "\nSignatureDef InputMap key :" + key +
                                "\nSignatureDef InputMap tfInfo: " + tfInfo.getName());
                    }
                }
            }



//            MetaGraphDef mgd = b.metaGraphDef();
//            List nodes = mgd.getGraphDef().getNodeList();
//            for (Object n : nodes){
//                System.out.println(nodes.toString());
//            }
//
//            Iterator<Operation> operations = b.graph().operations();
//            for (Iterator<Operation> it = operations; it.hasNext(); ) {
//                Operation o = it.next();
//                System.out.println(o.name());
//            }
//
//            List<Op> inits = b.graph().initializers();
//            for (Object i: inits){
//                System.out.println(inits.toString());
//            }

            List<Signature> signatures = bundle.signatures();
            for(Signature s : signatures){
                System.out.println("s.inputNames(): " + s.inputNames().toString());
                Set<String> inputNames = s.inputNames();
                for (Object sin : inputNames){
                    System.out.println("sin.toString(): " + sin.toString());
                }

                System.out.println("s.outputNames(): " + s.outputNames().toString());

                System.out.println("s.key(): " +s.key());
                System.out.println("s.methodName(): " +s.methodName());
            }

//            // create the session from the Bundle
//            Session sess = b.session();
//            // create an input Tensor, value = 2.0f
//            Tensor x = Tensor.create(
//                    new long[] {NUM_PREDICTIONS},
//                    FloatBuffer.wrap( new float[] {2.0f} )
//            );
//
//            // run the model and get the result, 4.0f.
//            float[] y = sess.runner()
//                    .feed("x", x)
//                    .fetch("y")
//                    .run()
//                    .get(0)
//                    .copyTo(new float[NUM_PREDICTIONS]);
//
//            // print out the result.
//            System.out.println(y[0]);
        }
    }
}