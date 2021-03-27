import org.tensorflow.*;
import org.tensorflow.op.Op;
import org.tensorflow.proto.framework.MetaGraphDef;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Iterator;
import java.util.List;

public class TFV1V2Reader {
    public static void main( String[] args ) throws IOException
    {
        // good idea to print the version number, 1.2.0 as of this writing
        System.out.println(TensorFlow.version());
        final int NUM_PREDICTIONS = 1;

        // load the model Bundle

//        String tf1ModelPath = "/home/wolliqeonii/workspace/dev/jsl/tf1-tests/saved_models/1616707109";
        String tf1ModelPath = "/home/wolliqeonii/Documents/jsl/tf2/bert1";

        try (SavedModelBundle b = SavedModelBundle.load(tf1ModelPath, "serve")) {

            System.out.println(b.metaGraphDef().hasGraphDef());

            MetaGraphDef mgd = b.metaGraphDef();
            List nodes = mgd.getGraphDef().getNodeList();
            for (Object n : nodes){
                System.out.println(nodes.toString());
            }

            Iterator<Operation> operations = b.graph().operations();
            for (Iterator<Operation> it = operations; it.hasNext(); ) {
                Operation o = it.next();
                System.out.println(o.name());
            }

            List<Op> inits = b.graph().initializers();
            for (Object i: inits){
                System.out.println(inits.toString());
            }

            List<Signature> signatures = b.signatures();
            for(Signature s : signatures){
                System.out.println(s.inputNames());
                System.out.println(s.outputNames());
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