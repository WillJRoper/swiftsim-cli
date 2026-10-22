/* Synthetic excerpt of the zoom branch's src/cell.h. */
enum cell_types {
  cell_type_regular, /* A standard top level cell. */
  cell_type_zoom,    /* A zoom cell. */
  cell_type_buff,    /* A buffer cell. */
  cell_type_bkg,     /* A background cell. */
} __attribute__((__packed__));

enum cell_subtypes {
  cell_subtype_regular,
  cell_subtype_neighbour,
  cell_subtype_void,
} __attribute__((__packed__));
