package labda.DDICorpus.xml;

import java.util.ArrayList;


public class ListEntity extends ArrayList<Entity> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4355025319961140271L;

	public boolean contains(Entity obj) {
		for (Entity ann:this) {
			if (ann.equalsText(obj) ||ann.equalsOffSet(obj)) return true;
		}
		return false;
	}
	
	public boolean overlap(Entity obj) {
		for (Entity ann:this) {
			if (ann.overlap(obj)) return true;
		}
		return false;
	}
}
