package labda.DDICorpus.xml;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;

/**
 * Class DocumentDDI.
 * 
 * @author Isabel Segura-Bedmar isegura@inf.uc3m.es
 * @date 	5 Marzo 2015
 */
public class DocumentDDI {

	

    /**
     * Field _id.
     */
    private String _id;

    
    /**
     * Field _sentenceList.
     */
    private List<Sentence> _sentenceList;
    
    
    
    //----------------/
    //- Constructors -/
   //----------------/

   public DocumentDDI() {
       super();
       this._sentenceList = new ArrayList<Sentence>();
   }


     //-----------/
    //- Methods -/
   //-----------/

   /**
    * 
    * 
    * @param vSentence
    * @throws IndexOutOfBoundsException if the index
    * given is outside the bounds of the collection
    */
   public void addSentence(
            Sentence vSentence)
   throws IndexOutOfBoundsException {
       this._sentenceList.add(vSentence);
   }

   /**
    * 
    * 
    * @param index
    * @param vSentence
    * @throws IndexOutOfBoundsException if the index
    * given is outside the bounds of the collection
    */
   public void addSentence(int index,Sentence vSentence)
   throws IndexOutOfBoundsException {
       this._sentenceList.add(index, vSentence);
   }
   
   /**
    * Returns the value of field 'id'.
    * 
    * @return the value of field 'Id'.
    */
   public String getId(
   ) {
       return this._id;
   }

   
   
   /**
    * Method getSentence.
    * 
    * @param index
    * @throws IndexOutOfBoundsException if the index
    * given is outside the bounds of the collection
    * @return the value of the Sentence at
    * the given index
    */
   public Sentence getSentence(int index)
   throws IndexOutOfBoundsException {
       // check bounds for index
       if (index < 0 || index >= this._sentenceList.size()) {
           throw new IndexOutOfBoundsException("getSentence: Index value '" + index + "' not in range [0.." + (this._sentenceList.size() - 1) + "]");
       }
       
       return (Sentence) _sentenceList.get(index);
   }

   /**
    * Method getSentence.Returns the contents of the collection in
    * an Array.  <p>Note:  Just in case the collection contents
    * are changing in another thread, we pass a 0-length Array of
    * the correct type into the API call.  This way we <i>know</i>
    * that the Array returned is of exactly the correct length.
    * 
    * @return this collection as an Array
    */
   public Sentence[] getSentences() {
       Sentence[] array = new Sentence[0];
       return (Sentence[]) this._sentenceList.toArray(array);
   }

   /**
    * Method getSentenceCount.
    * 
    * @return the size of this collection
    */
   public int getSentenceCount(
   ) {
       return this._sentenceList.size();
   }


   /**
    */
   public void removeAllSentence(
   ) {
       this._sentenceList.clear();
   }

   /**
    * Method removeSentence.
    * 
    * @param vSentence
    * @return true if the object was removed from the collection.
    */
   public boolean removeSentence(
           Sentence vSentence) {
       boolean removed = _sentenceList.remove(vSentence);
       return removed;
   }

   /**
    * Method removeSentenceAt.
    * 
    * @param index
    * @return the element removed from the collection
    */
   public Sentence removeSentenceAt(
            int index) {
       Object obj = this._sentenceList.remove(index);
       return (Sentence) obj;
   }

   /**
    * Sets the value of field 'id'.
    * 
    * @param id the value of field 'id'.
    */
   public void setId(
            String id) {
       this._id = id;
   }
   
   
   /**
    * 
    * 
    * @param index
    * @param vSentence
    * @throws java.lang.IndexOutOfBoundsException if the index
    * given is outside the bounds of the collection
    */
   public void setSentence(
           int index,
           Sentence vSentence)
   throws java.lang.IndexOutOfBoundsException {
       // check bounds for index
       if (index < 0 || index >= this._sentenceList.size()) {
           throw new IndexOutOfBoundsException("setSentence: Index value '" + index + "' not in range [0.." + (this._sentenceList.size() - 1) + "]");
       }
       
       this._sentenceList.set(index, vSentence);
   }

   /**
    * 
    * 
    * @param vSentenceArray
    */
   public void setSentence(
           Sentence[] vSentenceArray) {
       //-- copy array
       _sentenceList.clear();
       
       for (int i = 0; i < vSentenceArray.length; i++) {
               this._sentenceList.add(vSentenceArray[i]);
       }
   }

   public String toString() {
   	String str=this._id;
   	for (Sentence ent:this._sentenceList) {
   		str+="\n";
   		str+="\t"+ent.toString();
   	}
   	
   	str+="\n";
   	return str;
   }
   
   
   /**
    * Method unmarshal.
    * Reads the xml document and save their elements into the object DocumentDDI
    * @param sFile
    
    * @return the unmarshaled DocumentDDI
    */
   public static DocumentDDI unmarshal(String sFile){
	   DocumentDDI oDocDDI=null;
	   //System.out.println(sFile.substring(sFile.lastIndexOf("/")+1));
		//We create a SAXBuilder object ot parse the file xml
	    SAXBuilder builder = new SAXBuilder();
	    File xmlFile = new File( sFile );
	    try
	    {
	    	//document is created
	    	Document document = (Document) builder.build( xmlFile );
	        //we obtain the root: 'RowSet'
	        Element rootDoc = document.getRootElement();
	        String idDoc=rootDoc.getAttributeValue("id");
	        
	        oDocDDI=new DocumentDDI();
	        oDocDDI.setId(idDoc);

	        List lstXMLSentences = rootDoc.getChildren( "sentence" );
	        int size=lstXMLSentences.size();
	        if (size>0) { 
		        Sentence[] aSentence=new Sentence[size];
		           for ( int i = 0; i < size; i++ ){
		        	
		            Element sentence = (Element) lstXMLSentences.get(i);
		            String sId=sentence.getAttribute("id").getValue().toString();
		            String sText=sentence.getAttribute("text").getValue().toString();
		            Sentence obj=new Sentence();
		            obj.setId(sId);
		            obj.setText(sText);
		            //System.out.println(obj.toString());
		            
		            List<Element> lstEntity=sentence.getChildren("entity");
		            int sizeEntity=lstEntity.size();
		            for (int j=0;j<sizeEntity;j++) {

				            Element entity = (Element) lstEntity.get(j);
				            String sIdEntity=entity.getAttribute("id").getValue().toString();

				            String sTextEntity=entity.getAttribute("text").getValue().toString();
				            
				            String sTypeEntity=entity.getAttribute("type").getValue().toString();
				            String sOffsetEntity=entity.getAttribute("charOffset").getValue().toString();
				            
				            Entity oEntity=new Entity(sIdEntity,sTextEntity,sTypeEntity,sOffsetEntity);
//				            if (oEntity.getLocations()!=null && oEntity.getLocations().size()>1) {
//				            		System.out.println(oEntity.toString());
//				            	}
				            obj.addEntity(oEntity);
		            }
		            
		            
		            List lstPair=sentence.getChildren("pair");
		            int sizePair=lstPair.size();
		            for (int j=0;j<sizePair;j++) {
			            Element pair = (Element) lstPair.get(j);
			            String sIdPair=pair.getAttribute("id").getValue().toString();
			            String sE1=pair.getAttribute("e1").getValue().toString();
			            String sE2=pair.getAttribute("e2").getValue().toString();
			            String sDdi=pair.getAttribute("ddi").getValue().toString();
			            String sType=null;
			            try {
			            		sType=pair.getAttribute("type").getValue().toString();
			            	} catch (Exception e) {
			            	
			            }
			            Pair oPair=new Pair(sIdPair,sE1,sE2,sDdi,sType);
			            //System.out.println(oPair.toString());
			            
			            obj.addPair(oPair);
	            }
		            
		            
		            aSentence[i]=obj;
		            
		            
		            
		        }
		        oDocDDI.setSentence(aSentence);
	        }
	           
	           
	        
	    }catch ( IOException io ) {
	        System.out.println( io.getMessage() );
	    }catch ( JDOMException jdomex ) {
	        System.out.println( jdomex.getMessage() );
	    }
	    return oDocDDI;
	 
	}
   
	
	

}
