package labda.DDICorpus.xml;

public class Location {
	int _start;
	int _end;
	public Location(int _start, int _end) {
		this._start=_start;
		this._end=_end;
	}
	public int getStart() {
		return _start;
	}
	public void setStart(int _start) {
		this._start = _start;
	}
	public int getEnd() {
		return _end;
	}
	public void setEnd(int _end) {
		this._end = _end;
	}
}
