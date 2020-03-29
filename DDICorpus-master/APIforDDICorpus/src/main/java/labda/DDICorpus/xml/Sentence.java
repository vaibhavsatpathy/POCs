package labda.DDICorpus.xml;
import java.util.List;

/**
 * Sentence Entity.
 * 
 * @author Isabel Segura-Bedmar isegura@inf.uc3m.es
 * @date 	5 Marzo 2015
 */
public class Sentence {

	  /**
     * Field _id. id of the sentence
     */
    private String _id;

    /**
     * Field _text. text of  the sentence
     */
    private String _text;

    /**
     * Field _entityList. list of entities in the sentence
     */
    private List<Entity> _entityList;

    /**
     * Field _pairList. list of pairs of entities in the sentence
     */
    private List<Pair> _pairList;
    
    //----------------/
    //- Constructors -/
   //----------------/

   public Sentence() {
       super();
       this._entityList = new java.util.ArrayList<Entity>();
       this._pairList = new java.util.ArrayList<Pair>();
   }
   
   

   //-----------/
  //- Methods -/
 //-----------/

 /**
  * 
  * add an entity to the list of entities
  * @param vEntity
  */
 public void addEntity(Entity vEntity) {
     this._entityList.add(vEntity);
 }

 /**
  * 
  *   add an entity to the list of entities at the index position
  * @param index
  * @param vEntity
  * @throws IndexOutOfBoundsException if the index
  * given is outside the bounds of the collection
  */
 public void addEntity(
         int index,
         Entity vEntity)
 throws IndexOutOfBoundsException {
     this._entityList.add(index, vEntity);
 }

 /**
  * 
  * add a pair to the sentence
  * @param vPair
  */
 public void addPair(Pair vPair) {
     this._pairList.add(vPair);
 }

 /**
  * 
  * add a pair to the sentence, at the index position
  * @param index
  * @param vPair
  * @throws IndexOutOfBoundsException if the index
  * given is outside the bounds of the collection
  */
 public void addPair(int index,Pair vPair)
 throws IndexOutOfBoundsException {
     this._pairList.add(index, vPair);
 }
 
 /**
  * Method getEntity. Return the entity of the list in the index position
  * 
  * @param index
  * @throws IndexOutOfBoundsException if the index
  * given is outside the bounds of the collection
  * @return the value of the Entity at
  * the given index
  */
 public Entity getEntity(int index)
 throws IndexOutOfBoundsException {
     // check bounds for index
     if (index < 0 || index >= this._entityList.size()) {
         throw new IndexOutOfBoundsException("getEntity: Index value '" + index + "' not in range [0.." + (this._entityList.size() - 1) + "]");
     }
     
     return _entityList.get(index);
 }
 
 
 
 /**
  * Method getEntity.Returns the contents of the collection in
  * an Array.  <p>Note:  Just in case the collection contents
  * are changing in another thread, we pass a 0-length Array of
  * the correct type into the API call.  This way we <i>know</i>
  * that the Array returned is of exactly the correct length.
  * 
  * @return this collection as an Array
  */
 public Entity[] getEntity() {
     Entity[] array = new Entity[0];
     return (Entity[]) this._entityList.toArray(array);
 }
 
 /**
  * Method getEntityCount. Number of entities
  * 
  * @return the size of this collection
  */
 public int getEntityCount() {
     return this._entityList.size();
 }

 /**
  * Returns the value of field 'id'.
  * 
  * @return the value of field 'Id'.
  */
 public String getId() {
     return this._id;
 }
 
 
 /**
  * Method getPair.
  * 
  * @param index
  * @throws IndexOutOfBoundsException if the index
  * given is outside the bounds of the collection
  * @return the value of the Pair at the
  * given index
  */
 public Pair getPair(int index)
 throws IndexOutOfBoundsException {
     // check bounds for index
     if (index < 0 || index >= this._pairList.size()) {
         throw new IndexOutOfBoundsException("getPair: Index value '" + index + "' not in range [0.." + (this._pairList.size() - 1) + "]");
     }
     
     return _pairList.get(index);
 }

 
 /**
  * Method getPair.Returns the contents of the collection in an
  * Array.  <p>Note:  Just in case the collection contents are
  * changing in another thread, we pass a 0-length Array of the
  * correct type into the API call.  This way we <i>know</i>
  * that the Array returned is of exactly the correct length.
  * 
  * @return this collection as an Array
  */
 public Pair[] getPair() {
     Pair[] array = new Pair[0];
     return (Pair[]) this._pairList.toArray(array);
 }

 /**
  * Method getPairCount.
  * 
  * @return the size of this collection
  */
 public int getPairCount(
 ) {
     return this._pairList.size();
 }

 /**
  * Returns the value of field 'text'.
  * 
  * @return the value of field 'Text'.
  */
 public String getText(
 ) {
     return this._text;
 }
 
 
 /**
  */
 public void removeAllEntity(
 ) {
     this._entityList.clear();
 }

 /**
  */
 public void removeAllPair(
 ) {
     this._pairList.clear();
 }

 /**
  * Method removeEntity.
  * 
  * @param vEntity
  * @return true if the object was removed from the collection.
  */
 public boolean removeEntity(
          Entity vEntity) {
     boolean removed = _entityList.remove(vEntity);
     return removed;
 }

 /**
  * Method removeEntityAt.
  * 
  * @param index
  * @return the element removed from the collection
  */
 public Entity removeEntityAt(int index) {
     Object obj = this._entityList.remove(index);
     return (Entity) obj;
 }

 /**
  * Method removePair.
  * 
  * @param vPair
  * @return true if the object was removed from the collection.
  */
 public boolean removePair(Pair vPair) {
     boolean removed = _pairList.remove(vPair);
     return removed;
 }

 /**
  * Method removePairAt.
  * 
  * @param index
  * @return the element removed from the collection
  */
 public Pair removePairAt(int index) {
     Object obj = this._pairList.remove(index);
     return (Pair) obj;
 }
 
 /**
  * 
  * 
  * @param index
  * @param vEntity
  * @throws IndexOutOfBoundsException if the index
  * given is outside the bounds of the collection
  */
 public void setEntity(int index,Entity vEntity)
 throws IndexOutOfBoundsException {
     // check bounds for index
     if (index < 0 || index >= this._entityList.size()) {
         throw new IndexOutOfBoundsException("setEntity: Index value '" + index + "' not in range [0.." + (this._entityList.size() - 1) + "]");
     }
     
     this._entityList.set(index, vEntity);
 }

 /**
  * 
  * 
  * @param vEntityArray
  */
 public void setEntity(Entity[] vEntityArray) {
     //-- copy array
     _entityList.clear();
     
     for (int i = 0; i < vEntityArray.length; i++) {
             this._entityList.add(vEntityArray[i]);
     }
 }

 /**
  * Sets the value of field 'id'.
  * 
  * @param id the value of field 'id'.
  */
 public void setId(String id) {
     this._id = id;
 }
 

	 /**
     * 
     * 
     * @param index
     * @param vPair
     * @throws IndexOutOfBoundsException if the index
     * given is outside the bounds of the collection
     */
    public void setPair(
           int index,
           Pair vPair)
    throws IndexOutOfBoundsException {
        // check bounds for index
        if (index < 0 || index >= this._pairList.size()) {
            throw new IndexOutOfBoundsException("setPair: Index value '" + index + "' not in range [0.." + (this._pairList.size() - 1) + "]");
        }
        
        this._pairList.set(index, vPair);
    }

    /**
     * 
     * 
     * @param vPairArray
     */
    public void setPair(
           Pair[] vPairArray) {
        //-- copy array
        _pairList.clear();
        
        for (int i = 0; i < vPairArray.length; i++) {
                this._pairList.add(vPairArray[i]);
        }
    }

    /**
     * Sets the value of field 'text'.
     * 
     * @param text the value of field 'text'.
     */
    public void setText(String text) {
        this._text = text;
    }

    
    public String toString() {
    	String str=this._id+"\t"+this._text;
    	for (Entity ent:this._entityList) {
    		str+="\n";
    		str+="\t"+ent.toString();
    	}
    	str+="\n";
//    	for (Pair pair:this._pairList) {
//    		str+="\n";
//    		str+="\t"+pair.toString();
//    	}
//    	str+="\n";
    	return str;
    }
    
    /**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
}
