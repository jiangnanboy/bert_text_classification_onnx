package text_classification.bert_tokenizer;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Optional;

public class LoadModel {

    public static OrtSession session;
    public static OrtEnvironment env;
    /**
     * load model
     * @throws OrtException
     */
    public static void loadModel(String model_path) throws OrtException {
        System.out.println("load model...");
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(model_path, new OrtSession.SessionOptions());
    }

    /**
     * close model
     */
    public static void closeModel() {
        System.out.println("close model...");
        if (Optional.of(session).isPresent()) {
            try {
                session.close();
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }
        if(Optional.of(env).isPresent()) {
            env.close();
        }
    }

}
