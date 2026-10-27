#ifndef ALEX_TYPES_TD
#define ALEX_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Alex/AlexDialect.td"

class Alex_Type<string name, string typeMnemonic>
    : TypeDef<Alex_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Alex_BoolType : Alex_Type<"Bool", "bool"> {
  let summary = "Alex bool type";
  let description = [{
    An immutable boolean taking values 0 or 1, like Torch's '!torch.bool'.
  }];
}

#endif // ALEX_TYPES_TD
